//! The two-pass "refine" (hires-fix) render of the MiniMax H3 Ref2VA pipeline.
//!
//! A request that carries `refine` renders in two passes. The geometry is owned
//! by [`contract::RefinePlan`] (`mold-core`): scale, start index, pass-1 canvas
//! and forward counts all come from there, and this module never re-derives
//! them. It owns the tensor operations the plan needs.
//!
//! The request's width and height are the FINAL canvas. Pass 1 runs the whole
//! frozen sigma schedule on a canvas `1/scale` as large, producing a clean video
//! latent and a clean audio latent. The video latent is bilinearly upsampled per
//! frame (time unchanged, F32), both latents are re-noised to the sigma at the
//! plan's grid index with fresh noise from streams that cannot disturb pass 1,
//! and pass 2 denoises the FINAL canvas from that index to the end of the same
//! grid on the still-resident transformer, with the references and text
//! conditioning reused. Decode and mux see the final canvas only.
//!
//! Without a plan nothing in this module runs: the pass-1 noise draw order,
//! sigma grid, identities and ledger are exactly the single-pass ones.

use std::path::Path;

use anyhow::ensure;
use mold_candle::minimax_h3::LatentUpscaler;

use super::*;

/// Pass-2 video noise is drawn from `seed ^ this`, pass-2 audio noise from
/// `seed ^ REFINE_AUDIO_NOISE_SEED_XOR`: separate seeded streams (the LTX-2
/// stage-2 re-noise convention), so the pass-1 stream is never advanced.
const REFINE_VIDEO_NOISE_SEED_XOR: u64 = 0x4833_5245_4649_4e56; // "H3REFINV"
const REFINE_AUDIO_NOISE_SEED_XOR: u64 = 0x4833_5245_4649_4e41; // "H3REFINA"

/// The geometry pass 1 runs on: the plan's pass-1 canvas, with everything but
/// the spatial canvas (frames, audio length, mode) the final geometry's.
pub(crate) fn pass1_geometry(
    plan: &contract::RefinePlan,
    final_geometry: &H3Fl2VaGeometry,
) -> Result<H3Fl2VaGeometry> {
    let final_width = u32::try_from(final_geometry.width).context("H3 refine width")?;
    let final_height = u32::try_from(final_geometry.height).context("H3 refine height")?;
    let (width, height) = plan.pass1_canvas(final_width, final_height).ok_or_else(|| {
        anyhow!(
            "MiniMax H3 refine: final canvas {final_width}x{final_height} does not split into a pass-1 canvas on the {}-pixel grid",
            plan.scale * contract::VIDEO_ROW_STRIDE
        )
    })?;
    H3Fl2VaGeometry::from_canvas(
        final_geometry.mode,
        usize::try_from(width)?,
        usize::try_from(height)?,
        final_geometry.frames,
        0,
    )
}

/// Refuse a schedule or integrator the second pass cannot re-enter, and return
/// the number of pass-2 forwards.
pub(crate) fn pass2_forwards(
    plan: &contract::RefinePlan,
    schedule: &H3DualSchedule,
    sampler: H3SamplerKind,
) -> Result<usize> {
    ensure!(
        sampler.uses_euler_update(),
        "MiniMax H3 refine: the second pass supports Euler samplers only, not {}",
        sampler.as_str()
    );
    let forwards = schedule.counts().transformer_evaluations;
    plan.pass2_forwards(forwards).ok_or_else(|| {
        anyhow!(
            "MiniMax H3 refine: start index {} must address one of the {forwards} forwards",
            plan.start_index
        )
    })
}

/// `" free_vram_mib=N"` for a CUDA device (a cheap driver query taken at the
/// moment of the log line, not a peak: the operator reads the peak from
/// `nvidia-smi`), empty on any other device or if the query fails.
pub(crate) fn free_device_memory_note(device: &Device) -> String {
    match device.location() {
        candle_core::DeviceLocation::Cuda { gpu_id } => crate::device::free_vram_bytes(gpu_id)
            .map(|free| format!(" free_vram_mib={}", free >> 20))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// `dim_out` linear-interpolation weights over `dim_in` samples, half-pixel
/// centres with edge clamping (`align_corners=False`).
fn interpolation_matrix(dim_in: usize, dim_out: usize, device: &Device) -> Result<Tensor> {
    ensure!(dim_in > 0 && dim_out > 0, "empty interpolation axis");
    let ratio = dim_in as f64 / dim_out as f64;
    let mut weights = vec![0f32; dim_out * dim_in];
    for out in 0..dim_out {
        let source = ((out as f64 + 0.5) * ratio - 0.5).max(0.0);
        let low = (source.floor() as usize).min(dim_in - 1);
        let high = (low + 1).min(dim_in - 1);
        let fraction = (source - low as f64) as f32;
        weights[out * dim_in + low] += 1.0 - fraction;
        weights[out * dim_in + high] += fraction;
    }
    Tensor::from_vec(weights, (dim_out, dim_in), device).map_err(Into::into)
}

/// Bilinear upsample of an UNPATCHIFIED video latent `[1, C, T, H, W]` to
/// `[1, C, T, H * scale, W * scale]`: per frame, time untouched, in F32.
///
/// Interpolation is linear and every weight row sums to one, so it commutes
/// with the latent's per-channel affine normalisation: no denormalise /
/// renormalise round trip is needed.
pub(crate) fn upsample_video_latent(latent: &Tensor, scale: usize) -> Result<Tensor> {
    let (batch, channels, frames, height, width) = latent.dims5()?;
    ensure!(batch == 1, "refine upsample expects a singleton batch");
    let device = latent.device();
    let planes =
        latent
            .to_dtype(DType::F32)?
            .contiguous()?
            .reshape((channels * frames, height, width))?;
    let rows = interpolation_matrix(height, height * scale, device)?;
    let columns = interpolation_matrix(width, width * scale, device)?
        .t()?
        .contiguous()?;
    planes
        .broadcast_matmul(&columns)?
        .contiguous()?
        .transpose(1, 2)?
        .contiguous()?
        .broadcast_matmul(&rows.t()?.contiguous()?)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((1, channels, frames, height * scale, width * scale))
        .map_err(Into::into)
}

/// Engine-shaping switch for the refine pass's spatial upsample: the path of
/// the learned MiniMax H3 latent upscaler checkpoint. Unset keeps the bilinear
/// default.
pub(crate) const LATENT_UPSCALER_VARIABLE: &str = "MOLD_H3_LATENT_UPSCALER";

/// The refine pass's spatial upsample of a normalised video latent: the learned
/// upscaler when [`LATENT_UPSCALER_VARIABLE`] names its checkpoint, otherwise
/// [`upsample_video_latent`]. The learned path loads the checkpoint for each
/// call and runs it in BF16, the checkpoint's own precision.
pub(crate) fn upsample_for_refine(latent: &Tensor, scale: usize) -> Result<Tensor> {
    let Some(path) = crate::runtime_env::value(LATENT_UPSCALER_VARIABLE) else {
        return upsample_video_latent(latent, scale);
    };
    let (_, _, _, height, width) = latent.dims5()?;
    let upscaler = LatentUpscaler::load_file(Path::new(&path), latent.device(), DType::BF16)?;
    upscaler
        .forward(latent, scale as f64, height * scale, width * scale)
        .map_err(Into::into)
}

/// `sigma * noise + (1 - sigma) * clean`, in F32: the rectified-flow point at
/// noise level `sigma` on the line between clean data and `noise`.
pub(crate) fn renoise_at_sigma(clean: &Tensor, noise: &Tensor, sigma: f32) -> Result<Tensor> {
    ensure!(
        clean.dims() == noise.dims(),
        "refine re-noise shape mismatch: {:?} versus {:?}",
        clean.dims(),
        noise.dims()
    );
    ensure!(
        sigma.is_finite() && (0.0..=1.0).contains(&sigma),
        "refine re-noise sigma {sigma} is outside [0, 1]"
    );
    let sigma = f64::from(sigma);
    clean
        .to_dtype(DType::F32)?
        .affine(1.0 - sigma, 0.0)?
        .add(&noise.to_dtype(DType::F32)?.affine(sigma, 0.0)?)
        .map_err(Into::into)
}

/// The fresh pass-2 noise for the final canvas, drawn from two seeded streams
/// of their own (video first, then audio) so pass 1's stream is untouched.
pub(crate) fn draw_refine_noise(
    seed: u64,
    final_geometry: &H3Fl2VaGeometry,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let video = H3RequestNoise::new(seed ^ REFINE_VIDEO_NOISE_SEED_XOR).draw(
        "refine-video",
        0,
        &final_geometry.generated_video_shape(),
        device,
    )?;
    let audio = H3RequestNoise::new(seed ^ REFINE_AUDIO_NOISE_SEED_XOR).draw(
        "refine-audio",
        0,
        &final_geometry.generated_audio_row_shape(),
        device,
    )?;
    Ok((video, audio))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAN: contract::RefinePlan = contract::RefinePlan::PUBLISHED;

    fn euler_schedule() -> H3DualSchedule {
        H3DualSchedule::new_for_sampler_with_video_shift(
            9,
            H3SamplerKind::ComfyEuler,
            crate::minimax_h3::sampler::H3_VIDEO_SHIFT,
        )
        .unwrap()
    }

    /// The plan's start index is a point on the REAL 8-step shift-12 grid, not
    /// just on the formula `mold-core` derives it from: sigma 0.9231 for the
    /// video stream and 0.75 for the audio stream.
    #[test]
    fn the_plan_start_index_is_sigma_0_9231_on_the_schedule_the_tier_runs() {
        let schedule = euler_schedule();
        assert_eq!(schedule.counts().transformer_evaluations, 8);
        let (video, audio) = (
            schedule.video_sigmas()[PLAN.start_index],
            schedule.audio_sigmas()[PLAN.start_index],
        );
        assert!((video - 0.9231).abs() < 5e-5, "{video}");
        assert!((audio - 0.75).abs() < 1e-6, "{audio}");
    }

    // ----- EXPERIMENT (throwaway): the Ref2VA Turbo 4-step tier. -----
    // See docs/plans/h3-refine-4step-experiment.md.

    fn plan_4step() -> contract::RefinePlan {
        contract::RefinePlan::for_model_scale(contract::REF2VA_COMFY_TURBO_4STEP, 2).unwrap()
    }

    fn euler_schedule_4step() -> H3DualSchedule {
        H3DualSchedule::new_for_sampler_with_video_shift(
            5,
            H3SamplerKind::ComfyEuler,
            crate::minimax_h3::sampler::H3_VIDEO_SHIFT,
        )
        .unwrap()
    }

    /// Index 2 of the REAL 4-step grid is the same sigma as the 8-step tier's
    /// index 4: video 0.9231, audio 0.75; the grid is [1, .9730, .9231, .8, 0].
    #[test]
    fn the_four_step_plan_start_index_is_the_same_sigma_as_the_eight_step_one() {
        let schedule = euler_schedule_4step();
        let plan = plan_4step();
        assert_eq!(schedule.counts().transformer_evaluations, 4);
        assert_eq!(plan.start_index, 2);
        let video = schedule.video_sigmas();
        for (got, want) in video.iter().zip([1.0, 0.9730, 0.9231, 0.8, 0.0]) {
            assert!((got - want).abs() < 5e-5, "{video:?}");
        }
        assert!((video[plan.start_index] - 0.9231).abs() < 5e-5);
        assert!((schedule.audio_sigmas()[plan.start_index] - 0.75).abs() < 1e-6);
        let eight = euler_schedule();
        assert!(
            (video[plan.start_index] - eight.video_sigmas()[PLAN.start_index]).abs() < 1e-6,
            "both tiers re-enter at the same video sigma"
        );
    }

    #[test]
    fn the_four_step_second_pass_is_the_last_two_forwards_and_euler_only() {
        let euler = euler_schedule_4step();
        let plan = plan_4step();
        assert_eq!(
            pass2_forwards(&plan, &euler, H3SamplerKind::ComfyEuler).unwrap(),
            2
        );
        assert_eq!(plan.pass1_forwards(4), 4);
        assert_eq!(plan.total_forwards(4), Some(6));
        assert!(
            pass2_forwards(&plan, &euler, H3SamplerKind::ComfyResMultistep)
                .unwrap_err()
                .to_string()
                .contains("Euler")
        );
        // The 8-step plan does not fit the 4-forward grid (index 4 addresses no forward).
        assert!(pass2_forwards(&PLAN, &euler, H3SamplerKind::ComfyEuler).is_err());
    }

    #[test]
    fn pass1_geometry_is_the_plans_canvas_with_the_final_clip() {
        let final_geometry = H3Fl2VaGeometry::from_canvas(
            Mode::ReferenceToAudioVideo,
            1920,
            1088,
            contract::REVIEWED_COMPACT_FRAMES as usize,
            0,
        )
        .unwrap();
        let pass1 = pass1_geometry(&PLAN, &final_geometry).unwrap();
        assert_eq!((pass1.width, pass1.height), (960, 544));
        assert_eq!(pass1.frames, final_geometry.frames);
        assert_eq!(pass1.latent_frames, final_geometry.latent_frames);
        assert_eq!(
            pass1.audio_latents_per_channel,
            final_geometry.audio_latents_per_channel
        );
        // The plan, not this module, decides what splits: 1376 / 2 = 688 is off
        // the 32-pixel grid.
        let off_grid = H3Fl2VaGeometry::from_canvas(
            Mode::ReferenceToAudioVideo,
            1376,
            576,
            contract::REVIEWED_COMPACT_FRAMES as usize,
            0,
        )
        .unwrap();
        assert!(pass1_geometry(&PLAN, &off_grid)
            .unwrap_err()
            .to_string()
            .contains("does not split"));
    }

    #[test]
    fn the_second_pass_is_the_tail_of_the_grid_and_euler_only() {
        let euler = euler_schedule();
        assert_eq!(
            pass2_forwards(&PLAN, &euler, H3SamplerKind::ComfyEuler).unwrap(),
            4
        );
        assert_eq!(PLAN.total_forwards(8), Some(12));
        assert!(
            pass2_forwards(&PLAN, &euler, H3SamplerKind::ComfyResMultistep)
                .unwrap_err()
                .to_string()
                .contains("Euler")
        );
        let late = contract::RefinePlan {
            start_index: 8,
            ..PLAN
        };
        assert!(pass2_forwards(&late, &euler, H3SamplerKind::ComfyEuler).is_err());
    }

    fn ramp_latent(frames: usize, height: usize, width: usize) -> Tensor {
        // value = 10 * channel + 100 * frame + 2 * row + 3 * column: affine in
        // every axis, so bilinear interpolation must reproduce it exactly away
        // from the clamped edges.
        let mut values = Vec::new();
        for channel in 0..2 {
            for frame in 0..frames {
                for row in 0..height {
                    for column in 0..width {
                        values.push(
                            10.0 * channel as f32
                                + 100.0 * frame as f32
                                + 2.0 * row as f32
                                + 3.0 * column as f32,
                        );
                    }
                }
            }
        }
        Tensor::from_vec(values, (1, 2, frames, height, width), &Device::Cpu).unwrap()
    }

    #[test]
    fn upsampling_keeps_time_and_channels_and_scales_only_the_spatial_axes() {
        let latent = ramp_latent(3, 4, 5);
        let up = upsample_video_latent(&latent, 2).unwrap();
        assert_eq!(up.dims(), &[1, 2, 3, 8, 10]);
        assert_eq!(up.dtype(), DType::F32);
        // Frame/channel offsets survive: interpolation never mixes planes.
        let planes = up.reshape((6, 80)).unwrap().to_vec2::<f32>().unwrap();
        let mean = |plane: &Vec<f32>| plane.iter().sum::<f32>() / plane.len() as f32;
        let means: Vec<f32> = planes.iter().map(mean).collect();
        for (index, mean) in means.iter().enumerate() {
            let (channel, frame) = (index / 3, index % 3);
            let expected = 10.0 * channel as f32 + 100.0 * frame as f32 + 2.0 * 1.5 + 3.0 * 2.0;
            assert!(
                (mean - expected).abs() < 1e-3,
                "{index}: {mean} vs {expected}"
            );
        }
    }

    #[test]
    fn upsampling_is_half_pixel_bilinear_and_exact_on_a_ramp() {
        let latent = ramp_latent(1, 4, 5);
        let up = upsample_video_latent(&latent, 2).unwrap();
        let plane = up.narrow(1, 0, 1).unwrap().reshape((8, 10)).unwrap();
        let plane = plane.to_vec2::<f32>().unwrap();
        // Output pixel o samples the source at (o + 0.5) / 2 - 0.5; interior
        // rows/columns (source in [0, size - 1]) reproduce the affine ramp.
        for (row, values) in plane.iter().enumerate().take(7).skip(1) {
            for (column, value) in values.iter().enumerate().take(9).skip(1) {
                let expected = 2.0 * ((row as f32 + 0.5) / 2.0 - 0.5)
                    + 3.0 * ((column as f32 + 0.5) / 2.0 - 0.5);
                assert!(
                    (value - expected).abs() < 1e-4,
                    "({row},{column}) {value} vs {expected}"
                );
            }
        }
        // Edges clamp to the first/last sample instead of extrapolating.
        assert!((plane[0][0] - 0.0).abs() < 1e-5);
        assert!((plane[7][9] - (2.0 * 3.0 + 3.0 * 4.0)).abs() < 1e-5);
        // A constant field stays constant.
        let flat = Tensor::full(0.75f32, (1, 24, 2, 3, 3), &Device::Cpu).unwrap();
        let up = upsample_video_latent(&flat, 3).unwrap();
        assert_eq!(up.dims(), &[1, 24, 2, 9, 9]);
        for value in up.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!((value - 0.75).abs() < 1e-6);
        }
    }

    #[test]
    fn renoise_is_the_flow_matching_line_between_clean_and_noise() {
        let clean = Tensor::new(&[1.0f32, -2.0, 4.0], &Device::Cpu).unwrap();
        let noise = Tensor::new(&[3.0f32, 2.0, 0.0], &Device::Cpu).unwrap();
        let at = |sigma| {
            renoise_at_sigma(&clean, &noise, sigma)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(at(0.0), vec![1.0, -2.0, 4.0]);
        assert_eq!(at(1.0), vec![3.0, 2.0, 0.0]);
        let half = at(0.5);
        for (value, expected) in half.iter().zip([2.0, 0.0, 2.0]) {
            assert!((value - expected).abs() < 1e-6);
        }
        assert!(renoise_at_sigma(&clean, &noise, 1.5).is_err());
        assert!(renoise_at_sigma(
            &clean,
            &Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap(),
            0.5
        )
        .is_err());
    }
}
