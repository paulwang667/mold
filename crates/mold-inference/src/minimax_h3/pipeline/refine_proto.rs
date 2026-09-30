//! RESEARCH PROTOTYPE, env-gated, NOT a product surface: a "hires-fix" second
//! pass for the MiniMax H3 Ref2VA pipeline.
//!
//! `MOLD_H3_REFINE_PROTO` is unset (or `0`/`off`) in every production run, and
//! then nothing in this module runs: the pass-1 noise draw order, sigma grid,
//! identities and ledger are exactly the ones the gate-off code path always
//! had. There is no request field, no admission arm, no budget line and no
//! provenance field for it; admission still prices the FINAL canvas.
//!
//! # What the gate does
//!
//! The request's width and height are the FINAL canvas. Pass 1 runs the whole
//! frozen sigma schedule on a canvas `1/scale` as large (each axis a multiple
//! of 32, else the request is refused), producing a clean video latent and a
//! clean audio latent. The video latent is bilinearly upsampled per frame
//! (time unchanged), both latents are re-noised to the sigma at grid index
//! `start` with fresh noise from streams that cannot disturb pass 1, and pass 2
//! denoises the FINAL canvas from `start` to the end of the same grid. Decode
//! and mux see the final canvas only.
//!
//! ```text
//! MOLD_H3_REFINE_PROTO=scale=2,start=4,lora2=0.5
//! ```
//!
//! - `scale`: spatial factor, default `2` (at least `2`).
//! - `start`: index into the frozen sigma grid where pass 2 begins, default
//!   `4` (sigma 0.923 on the 8-step shift-12 Turbo grid). Must address an
//!   existing forward; `0` re-denoises from pure noise.
//! - `lora2`: Turbo LoRA strength for pass 2 only, in `(0, 1]`; absent means
//!   the request's strength is kept.
//! - `uncap`: `1`/`true`/`on` lifts the request-side canvas AREA ceilings (the
//!   compact rule, the family `MAX_PIXELS`, the profile's `max_pixels`) to
//!   `mold_core::minimax_h3::UNCAP_MAX_PIXELS` (4 Mi) and waives the memory
//!   refusals whose grants are extrapolated from the 1344x768 measurement, so a
//!   FINAL canvas such as 1920x832, 2560x1088 or 2688x1536 can be asked for.
//!   `mold-core` cannot see this module, so it re-reads the same variable for
//!   this one key only (`refine_proto_uncap_from_spec`); this parser stays the
//!   single owner of `scale`, `start` and `lora2`. Alignment, minimum axis and
//!   aspect rules are unchanged, and pass 1 gets no new restriction.
//!
//! A bare `1`/`on`/`true` selects the defaults; empty, `0`, `off` and `false`
//! leave the gate closed.

use anyhow::ensure;

use super::*;

/// Registered engine-shaping variable (`runtime_env::ENGINE_SHAPING_VARIABLES`).
pub(crate) const REFINE_PROTO_VARIABLE: &str = "MOLD_H3_REFINE_PROTO";

/// Pass-2 video noise is drawn from `seed ^ this`, pass-2 audio noise from
/// `seed ^ REFINE_AUDIO_NOISE_SEED_XOR`: separate seeded streams (the LTX-2
/// stage-2 re-noise convention), so the pass-1 stream is never advanced.
const REFINE_VIDEO_NOISE_SEED_XOR: u64 = 0x4833_5245_4649_4e56; // "H3REFINV"
const REFINE_AUDIO_NOISE_SEED_XOR: u64 = 0x4833_5245_4649_4e41; // "H3REFINA"

/// The 768p canvas is composed of 32-pixel units (16x VAE times 2x patch).
const CANVAS_UNIT: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct H3RefineProto {
    pub scale: usize,
    pub start: usize,
    pub lora2: Option<f32>,
    /// Lift the request-side canvas area ceilings (see the module docs). The
    /// admission code reads it through `mold-core`; it rides here so `parse`
    /// accepts the key and the two readers can be checked against each other.
    pub uncap: bool,
}

impl H3RefineProto {
    pub(crate) const DEFAULT_SCALE: usize = 2;
    pub(crate) const DEFAULT_START: usize = 4;

    /// Read the gate from the process-frozen runtime environment.
    pub(crate) fn from_environment() -> Result<Option<Self>> {
        match crate::runtime_env::value(REFINE_PROTO_VARIABLE) {
            Some(spec) => Self::parse(&spec),
            None => Ok(None),
        }
    }

    pub(crate) fn parse(spec: &str) -> Result<Option<Self>> {
        let spec = spec.trim();
        match spec.to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" => return Ok(None),
            "1" | "on" | "true" => return Ok(Some(Self::defaults())),
            _ => {}
        }
        let mut parsed = Self::defaults();
        let mut seen = Vec::new();
        for item in spec.split(',') {
            let (key, value) = item.split_once('=').ok_or_else(|| {
                anyhow!("{REFINE_PROTO_VARIABLE}: expected key=value, got {item:?}")
            })?;
            let (key, value) = (key.trim(), value.trim());
            ensure!(
                !seen.contains(&key),
                "{REFINE_PROTO_VARIABLE}: {key} given twice"
            );
            seen.push(key);
            match key {
                "scale" => {
                    parsed.scale = value
                        .parse()
                        .with_context(|| format!("{REFINE_PROTO_VARIABLE}: scale {value:?}"))?;
                    ensure!(
                        parsed.scale >= 2,
                        "{REFINE_PROTO_VARIABLE}: scale must be an integer of at least 2"
                    );
                }
                "start" => {
                    parsed.start = value
                        .parse()
                        .with_context(|| format!("{REFINE_PROTO_VARIABLE}: start {value:?}"))?;
                }
                "lora2" => {
                    let strength: f32 = value
                        .parse()
                        .with_context(|| format!("{REFINE_PROTO_VARIABLE}: lora2 {value:?}"))?;
                    ensure!(
                        strength.is_finite() && strength > 0.0 && strength <= 1.0,
                        "{REFINE_PROTO_VARIABLE}: lora2 must be in (0, 1]"
                    );
                    parsed.lora2 = Some(strength);
                }
                "uncap" => {
                    parsed.uncap = match value.to_ascii_lowercase().as_str() {
                        "1" | "true" | "on" => true,
                        "0" | "false" | "off" => false,
                        _ => bail!("{REFINE_PROTO_VARIABLE}: uncap must be 0 or 1, got {value:?}"),
                    };
                }
                other => bail!("{REFINE_PROTO_VARIABLE}: unknown key {other:?}"),
            }
        }
        Ok(Some(parsed))
    }

    fn defaults() -> Self {
        Self {
            scale: Self::DEFAULT_SCALE,
            start: Self::DEFAULT_START,
            lora2: None,
            uncap: false,
        }
    }

    /// Pass-1 canvas for a final canvas: each axis divided by `scale`, which
    /// must leave a multiple of 32.
    pub(crate) fn pass1_canvas(&self, width: usize, height: usize) -> Result<(usize, usize)> {
        let divide = |axis: &str, value: usize| -> Result<usize> {
            let pass1 = value / self.scale;
            ensure!(
                value.is_multiple_of(self.scale) && pass1 > 0 && pass1.is_multiple_of(CANVAS_UNIT),
                "{REFINE_PROTO_VARIABLE}: final {axis} {value} / scale {} = {} is not a multiple of {CANVAS_UNIT}",
                self.scale,
                value as f64 / self.scale as f64
            );
            Ok(pass1)
        };
        Ok((divide("width", width)?, divide("height", height)?))
    }

    /// The geometry pass 1 runs on. Everything but the spatial canvas (frames,
    /// audio length, mode) is the final geometry's.
    pub(crate) fn pass1_geometry(
        &self,
        final_geometry: &H3Fl2VaGeometry,
    ) -> Result<H3Fl2VaGeometry> {
        let (width, height) = self.pass1_canvas(final_geometry.width, final_geometry.height)?;
        H3Fl2VaGeometry::from_canvas(final_geometry.mode, width, height, final_geometry.frames, 0)
    }

    /// Refuse a schedule or integrator the prototype cannot re-enter, and
    /// return the number of pass-2 forwards.
    pub(crate) fn pass2_forwards(
        &self,
        schedule: &H3DualSchedule,
        sampler: H3SamplerKind,
    ) -> Result<usize> {
        ensure!(
            sampler.uses_euler_update(),
            "{REFINE_PROTO_VARIABLE}: the second pass supports Euler samplers only, not {}",
            sampler.as_str()
        );
        let forwards = schedule.counts().transformer_evaluations;
        ensure!(
            self.start < forwards,
            "{REFINE_PROTO_VARIABLE}: start {} must address one of the {forwards} forwards",
            self.start
        );
        Ok(forwards - self.start)
    }

    /// Total coupled forwards the phase ledger must expect: pass 1's full
    /// schedule plus pass 2's tail.
    pub(crate) fn total_forwards(&self, pass1_forwards: usize) -> Result<usize> {
        ensure!(
            self.start < pass1_forwards,
            "{REFINE_PROTO_VARIABLE}: start {} must address one of the {pass1_forwards} forwards",
            self.start
        );
        Ok(pass1_forwards + (pass1_forwards - self.start))
    }
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

    fn refine(spec: &str) -> Option<H3RefineProto> {
        H3RefineProto::parse(spec).unwrap()
    }

    #[test]
    fn the_gate_is_closed_unless_asked_for() {
        for closed in ["", "  ", "0", "off", "OFF", "false"] {
            assert_eq!(refine(closed), None, "{closed:?}");
        }
        for open in ["1", "on", "true"] {
            assert_eq!(
                refine(open),
                Some(H3RefineProto {
                    scale: 2,
                    start: 4,
                    lora2: None,
                    uncap: false,
                })
            );
        }
    }

    #[test]
    fn a_key_list_overrides_defaults_and_rejects_anything_unclear() {
        assert_eq!(
            refine("scale=3, start=2 ,lora2=0.5"),
            Some(H3RefineProto {
                scale: 3,
                start: 2,
                lora2: Some(0.5),
                uncap: false,
            })
        );
        assert_eq!(
            refine("start=6"),
            Some(H3RefineProto {
                scale: 2,
                start: 6,
                lora2: None,
                uncap: false,
            })
        );
        for bad in [
            "scale=1",
            "scale=two",
            "start=-1",
            "lora2=0",
            "lora2=1.5",
            "lora2=nan",
            "scale=2,scale=3",
            "shift=2",
            "scale",
            "scale=2,,start=1",
        ] {
            assert!(
                H3RefineProto::parse(bad).is_err(),
                "{bad:?} must be refused"
            );
        }
    }

    #[test]
    fn uncap_is_a_boolean_key_and_the_two_readers_agree_on_it() {
        assert!(!refine("scale=2").unwrap().uncap);
        assert!(!refine("1").unwrap().uncap);
        for spec in [
            "uncap=1",
            "scale=2,start=4,uncap=1",
            "uncap=true,lora2=0.5",
            "uncap=on",
        ] {
            assert!(refine(spec).unwrap().uncap, "{spec:?}");
            assert!(mold_core::minimax_h3::refine_proto_uncap_from_spec(spec));
        }
        for spec in ["uncap=0", "uncap=off", "scale=2,uncap=false", "1", "on", ""] {
            if let Some(parsed) = H3RefineProto::parse(spec).unwrap() {
                assert!(!parsed.uncap, "{spec:?}");
            }
            assert!(!mold_core::minimax_h3::refine_proto_uncap_from_spec(spec));
        }
        assert!(H3RefineProto::parse("uncap=maybe").is_err());
        assert!(H3RefineProto::parse("uncap=1,uncap=0").is_err());
    }

    #[test]
    fn the_uncapped_ladder_splits_into_pass_one_canvases_in_32_multiples() {
        let proto = refine("scale=2,uncap=1").unwrap();
        for (final_canvas, pass1) in [
            ((1920, 832), (960, 416)),
            ((2560, 1088), (1280, 544)),
            ((2688, 1536), (1344, 768)),
        ] {
            assert_eq!(
                proto.pass1_canvas(final_canvas.0, final_canvas.1).unwrap(),
                pass1
            );
        }
    }

    #[test]
    fn pass1_canvas_divides_the_final_canvas_into_32_multiples_or_refuses() {
        let proto = refine("scale=2").unwrap();
        assert_eq!(proto.pass1_canvas(1344, 576).unwrap(), (672, 288));
        // 1344 / 2 = 672 = 21 * 32 fine; 1376 / 2 = 688 is not a multiple of 32.
        assert!(proto.pass1_canvas(1376, 576).is_err());
        assert!(proto.pass1_canvas(1344, 544).is_err());
        assert!(proto.pass1_canvas(1345, 576).is_err());
        assert!(refine("scale=5").unwrap().pass1_canvas(1344, 576).is_err());
        assert_eq!(
            refine("scale=3").unwrap().pass1_canvas(1728, 576).unwrap(),
            (576, 192)
        );
    }

    #[test]
    fn the_ledger_count_is_pass_one_plus_the_tail_from_start() {
        let proto = refine("start=4").unwrap();
        assert_eq!(proto.total_forwards(8).unwrap(), 12);
        assert_eq!(refine("start=0").unwrap().total_forwards(8).unwrap(), 16);
        assert_eq!(refine("start=7").unwrap().total_forwards(8).unwrap(), 9);
        assert!(refine("start=8").unwrap().total_forwards(8).is_err());
    }

    #[test]
    fn only_euler_schedules_can_be_re_entered() {
        let euler = H3DualSchedule::new_for_sampler_with_video_shift(
            9,
            H3SamplerKind::ComfyEuler,
            crate::minimax_h3::sampler::H3_VIDEO_SHIFT,
        )
        .unwrap();
        let proto = refine("start=4").unwrap();
        assert_eq!(
            proto
                .pass2_forwards(&euler, H3SamplerKind::ComfyEuler)
                .unwrap(),
            4
        );
        assert!(proto
            .pass2_forwards(&euler, H3SamplerKind::ComfyResMultistep)
            .unwrap_err()
            .to_string()
            .contains("Euler"));
        assert!(refine("start=8")
            .unwrap()
            .pass2_forwards(&euler, H3SamplerKind::ComfyEuler)
            .is_err());
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
