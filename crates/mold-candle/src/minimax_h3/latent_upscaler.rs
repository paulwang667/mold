//! Learned spatial upscaler for the MiniMax H3 refine pass.
//!
//! A port of the 3D latent resizer shipped by the ComfyUI node
//! `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler` (release
//! `minimax_h3_latent_upscaler_3d_conv_v1`, Apache-2.0, 345,280,216 parameters).
//! The architecture and its key layout are read from the node source and the
//! released safetensors header; `tests` pins the port against the node's own
//! float32 output (`testdata/minimax_h3/latent_upscaler_reference_t4h3w5_s2.safetensors`).
//!
//! The node normalises its input with the H3 latent statistics and
//! denormalises its output with the same statistics. Mold's refine latents are
//! already normalised, so those two steps cancel and the network runs directly
//! on them: `z_out = net(z_in, scale)`.
//!
//! The network is non-causal, so each 3x3x3 convolution is evaluated as one
//! 2-D convolution per temporal tap and the taps are summed with zero
//! padding along time. The one departure from the node is that a clip is
//! never split into temporal chunks: the node chunks only above 32 latent
//! frames, and its overlap blending is not reproduced.

use std::path::Path;

use candle::{DType, Device, IndexOp, Result, Tensor, bail};
use candle_nn::{Linear, Module, VarBuilder, linear, ops};

/// Latent channels the released network consumes and produces (`conv_in` and
/// `conv_out` are 24-channel).
pub const LATENT_UPSCALER_CHANNELS: usize = 24;
const MODEL_CHANNELS: usize = 512;
const GROUPS: usize = 32;
const EPS: f64 = 1e-5;
const EMBED_DIM: usize = 64;
const RES_BLOCKS_PER_SIDE: usize = 12;
const TEMPORAL_EVERY: usize = 2;
const TEMPORAL_KERNEL: usize = 5;

/// One entry of the interleaved block stack the node builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind {
    Res,
    Temporal,
}

/// The node's block order: every residual block is followed by a temporal
/// block after each even index. Twelve residual blocks give eighteen modules
/// per side, and the released keys hold exactly that many.
fn block_plan() -> Vec<BlockKind> {
    let mut plan = Vec::with_capacity(RES_BLOCKS_PER_SIDE * 2);
    for block in 0..RES_BLOCKS_PER_SIDE {
        plan.push(BlockKind::Res);
        if block % TEMPORAL_EVERY == 0 {
            plan.push(BlockKind::Temporal);
        }
    }
    plan
}

/// Reads one tensor, detaching it from the mmap on CPU the way the H3 VAE does.
fn take(vb: &VarBuilder, shape: impl Into<candle::Shape>, name: &str) -> Result<Tensor> {
    let tensor = vb.get(shape, name)?;
    if tensor.device().is_cpu() {
        tensor.copy()
    } else {
        Ok(tensor)
    }
}

/// `dim` shifted by `shift` with zeros entering at the boundary:
/// `out[t] = x[t + shift]`, or zero where that index leaves the range.
fn shift_along(x: &Tensor, dim: usize, shift: isize) -> Result<Tensor> {
    if shift == 0 {
        return Ok(x.clone());
    }
    let len = x.dim(dim)?;
    let taken = shift.unsigned_abs();
    if taken >= len {
        return x.zeros_like();
    }
    let mut zero_dims = x.dims().to_vec();
    zero_dims[dim] = taken;
    let zeros = Tensor::zeros(zero_dims, x.dtype(), x.device())?;
    if shift > 0 {
        let kept = x.narrow(dim, taken, len - taken)?;
        Tensor::cat(&[&kept, &zeros], dim)
    } else {
        let kept = x.narrow(dim, 0, len - taken)?;
        Tensor::cat(&[&zeros, &kept], dim)
    }
}

/// GroupNorm(32) over `[B, C, T, H, W]`, computed in float32.
#[derive(Clone, Debug)]
struct GroupNorm3d {
    weight: Tensor,
    bias: Tensor,
}

impl GroupNorm3d {
    fn load(vb: &VarBuilder, channels: usize) -> Result<Self> {
        Ok(Self {
            weight: take(vb, channels, "weight")?,
            bias: take(vb, channels, "bias")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, height, width) = x.dims5()?;
        let dtype = x.dtype();
        let per_group = channels / GROUPS;
        let grouped = x.to_dtype(DType::F32)?.contiguous()?.reshape((
            batch,
            GROUPS,
            per_group * frames * height * width,
        ))?;
        let mean = grouped.mean_keepdim(2)?;
        let centred = grouped.broadcast_sub(&mean)?;
        let variance = centred.sqr()?.mean_keepdim(2)?;
        let normalised = centred
            .broadcast_div(&variance.affine(1.0, EPS)?.sqrt()?)?
            .reshape((batch, channels, frames, height, width))?;
        let weight = self
            .weight
            .to_dtype(DType::F32)?
            .reshape((1, channels, 1, 1, 1))?;
        let bias = self
            .bias
            .to_dtype(DType::F32)?
            .reshape((1, channels, 1, 1, 1))?;
        normalised
            .broadcast_mul(&weight)?
            .broadcast_add(&bias)?
            .to_dtype(dtype)
    }
}

/// A 3-D convolution evaluated as one 2-D convolution per temporal tap.
#[derive(Clone, Debug)]
struct Conv3d {
    /// One `[out, in, kh, kw]` slice of the weight per temporal tap.
    taps: Vec<Tensor>,
    bias: Tensor,
    temporal_padding: usize,
    spatial_padding: usize,
}

impl Conv3d {
    fn load(vb: &VarBuilder, input: usize, output: usize, kernel: [usize; 3]) -> Result<Self> {
        let [kt, kh, kw] = kernel;
        let weight = take(vb, (output, input, kt, kh, kw), "weight")?;
        let bias = take(vb, output, "bias")?;
        let taps = (0..kt)
            .map(|tap| weight.i((.., .., tap, .., ..))?.contiguous())
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            taps,
            bias,
            temporal_padding: kt / 2,
            spatial_padding: kh.max(kw) / 2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, height, width) = x.dims5()?;
        let out_channels = self.bias.dim(0)?;
        let images = x.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((
            batch * frames,
            channels,
            height,
            width,
        ))?;
        let mut acc: Option<Tensor> = None;
        for (tap, kernel) in self.taps.iter().enumerate() {
            let offset = tap as isize - self.temporal_padding as isize;
            let conv = images
                .conv2d(kernel, self.spatial_padding, 1, 1, 1)?
                .reshape((batch, frames, out_channels, height, width))?;
            let shifted = shift_along(&conv, 1, offset)?;
            acc = Some(match acc {
                None => shifted,
                Some(sum) => (sum + shifted)?,
            });
        }
        let Some(sum) = acc else {
            bail!("MiniMax H3 latent upscaler convolution has no temporal taps");
        };
        sum.broadcast_add(&self.bias.reshape((1, 1, out_channels, 1, 1))?)?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()
    }
}

/// Depthwise temporal convolution followed by a pointwise projection, with a
/// residual connection.
#[derive(Clone, Debug)]
struct TemporalConv {
    norm: GroupNorm3d,
    /// One `[1, C, 1, 1, 1]` gain per temporal tap.
    taps: Vec<Tensor>,
    bias: Tensor,
    pointwise: Conv3d,
}

impl TemporalConv {
    fn load(vb: &VarBuilder, channels: usize) -> Result<Self> {
        let weight = take(
            &vb.pp("dwconv"),
            (channels, 1, TEMPORAL_KERNEL, 1, 1),
            "weight",
        )?;
        let taps = (0..TEMPORAL_KERNEL)
            .map(|tap| {
                weight
                    .i((.., 0, tap, 0, 0))?
                    .reshape((1, channels, 1, 1, 1))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            norm: GroupNorm3d::load(&vb.pp("norm"), channels)?,
            taps,
            bias: take(&vb.pp("dwconv"), channels, "bias")?,
            pointwise: Conv3d::load(&vb.pp("pwconv"), channels, channels, [1, 1, 1])?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = ops::silu(&self.norm.forward(x)?)?;
        let padding = (TEMPORAL_KERNEL / 2) as isize;
        let mut acc: Option<Tensor> = None;
        for (tap, gain) in self.taps.iter().enumerate() {
            let shifted = shift_along(&hidden, 2, tap as isize - padding)?.broadcast_mul(gain)?;
            acc = Some(match acc {
                None => shifted,
                Some(sum) => (sum + shifted)?,
            });
        }
        let Some(sum) = acc else {
            bail!("MiniMax H3 latent upscaler temporal block has no taps");
        };
        let channels = self.bias.dim(0)?;
        let depthwise = sum.broadcast_add(&self.bias.reshape((1, channels, 1, 1, 1))?)?;
        Ok((x + self.pointwise.forward(&depthwise)?)?)
    }
}

/// Residual block with a scale/shift modulation from the scale embedding.
#[derive(Clone, Debug)]
struct ResBlock {
    channels: usize,
    in_norm: GroupNorm3d,
    in_conv: Conv3d,
    emb: Linear,
    out_norm: GroupNorm3d,
    out_conv: Conv3d,
}

impl ResBlock {
    fn load(vb: &VarBuilder, channels: usize) -> Result<Self> {
        Ok(Self {
            channels,
            in_norm: GroupNorm3d::load(&vb.pp("in_layers.0"), channels)?,
            in_conv: Conv3d::load(&vb.pp("in_layers.2"), channels, channels, [3, 3, 3])?,
            emb: linear(EMBED_DIM, 2 * channels, vb.pp("emb_layers.1"))?,
            out_norm: GroupNorm3d::load(&vb.pp("out_norm"), channels)?,
            out_conv: Conv3d::load(&vb.pp("out_layers.2"), channels, channels, [3, 3, 3])?,
        })
    }

    fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let hidden = self
            .in_conv
            .forward(&ops::silu(&self.in_norm.forward(x)?)?)?;
        let modulation = ops::silu(emb)?.apply(&self.emb)?.to_dtype(hidden.dtype())?;
        let scale = modulation
            .narrow(1, 0, self.channels)?
            .reshape((1, self.channels, 1, 1, 1))?;
        let shift = modulation
            .narrow(1, self.channels, self.channels)?
            .reshape((1, self.channels, 1, 1, 1))?;
        let modulated = self
            .out_norm
            .forward(&hidden)?
            .broadcast_mul(&scale.affine(1.0, 1.0)?)?
            .broadcast_add(&shift)?;
        let out = self.out_conv.forward(&ops::silu(&modulated)?)?;
        Ok((x + out)?)
    }
}

#[derive(Clone, Debug)]
enum Block {
    Res(ResBlock),
    Temporal(TemporalConv),
}

impl Block {
    fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        match self {
            Self::Res(block) => block.forward(x, emb),
            Self::Temporal(block) => block.forward(x),
        }
    }
}

fn load_blocks(vb: &VarBuilder, side: &str) -> Result<Vec<Block>> {
    block_plan()
        .into_iter()
        .enumerate()
        .map(|(index, kind)| {
            let vb = vb.pp(side).pp(index.to_string());
            Ok(match kind {
                BlockKind::Res => Block::Res(ResBlock::load(&vb, MODEL_CHANNELS)?),
                BlockKind::Temporal => Block::Temporal(TemporalConv::load(&vb, MODEL_CHANNELS)?),
            })
        })
        .collect()
}

/// Half-pixel linear interpolation weights (`align_corners=False`), as
/// `[dim_out, dim_in]`. Same convention as the refine pass's bilinear default.
fn interpolation_matrix(dim_in: usize, dim_out: usize, device: &Device) -> Result<Tensor> {
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
    Tensor::from_vec(weights, (dim_out, dim_in), device)
}

/// Spatial resize of `[B, C, T, H, W]` to `height x width`, time untouched.
fn resize_spatial(x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (batch, channels, frames, in_height, in_width) = x.dims5()?;
    let dtype = x.dtype();
    let device = x.device();
    let rows = interpolation_matrix(in_height, height, device)?;
    let columns = interpolation_matrix(in_width, width, device)?
        .t()?
        .contiguous()?;
    x.to_dtype(DType::F32)?
        .contiguous()?
        .reshape((batch * channels * frames, in_height, in_width))?
        .broadcast_matmul(&columns)?
        .contiguous()?
        .transpose(1, 2)?
        .contiguous()?
        .broadcast_matmul(&rows.t()?.contiguous()?)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch, channels, frames, height, width))?
        .to_dtype(dtype)
}

/// The released 3-D latent upscaler.
#[derive(Clone, Debug)]
pub struct LatentUpscaler {
    embed_in: Linear,
    embed_out: Linear,
    conv_in: Conv3d,
    in_blocks: Vec<Block>,
    out_blocks: Vec<Block>,
    norm_out: GroupNorm3d,
    conv_out: Conv3d,
}

impl LatentUpscaler {
    /// Loads from a `VarBuilder` rooted at the checkpoint's own key names.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embed_in: linear(1, EMBED_DIM, vb.pp("embed.0"))?,
            embed_out: linear(EMBED_DIM, EMBED_DIM, vb.pp("embed.2"))?,
            conv_in: Conv3d::load(
                &vb.pp("conv_in"),
                LATENT_UPSCALER_CHANNELS,
                MODEL_CHANNELS,
                [3, 3, 3],
            )?,
            in_blocks: load_blocks(&vb, "in_blocks")?,
            out_blocks: load_blocks(&vb, "out_blocks")?,
            norm_out: GroupNorm3d::load(&vb.pp("norm_out"), MODEL_CHANNELS)?,
            conv_out: Conv3d::load(
                &vb.pp("conv_out"),
                MODEL_CHANNELS,
                LATENT_UPSCALER_CHANNELS,
                [3, 3, 3],
            )?,
        })
    }

    /// Loads the released safetensors file, converting weights to `dtype`.
    pub fn load_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
        Self::load(vb)
    }

    /// Upscales a normalised latent `[1, 24, T, H, W]` to `[1, 24, T, height, width]`.
    /// `scale` is the spatial factor the scale embedding is conditioned on.
    pub fn forward(
        &self,
        latent: &Tensor,
        scale: f64,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let (batch, channels, _frames, in_height, in_width) = latent.dims5()?;
        if batch != 1 {
            bail!("MiniMax H3 latent upscaler expects a singleton batch, got {batch}");
        }
        if channels != LATENT_UPSCALER_CHANNELS {
            bail!(
                "MiniMax H3 latent upscaler expects {LATENT_UPSCALER_CHANNELS} channels, got {channels}"
            );
        }
        if (height, width) == (in_height, in_width) {
            return Ok(latent.clone());
        }
        let dtype = self.conv_in.bias.dtype();
        let scale_input = Tensor::from_slice(&[(scale - 1.0) as f32], (1, 1), latent.device())?
            .to_dtype(dtype)?;
        let emb = self
            .embed_out
            .forward(&ops::silu(&self.embed_in.forward(&scale_input)?)?)?;
        let mut x = self.conv_in.forward(&latent.to_dtype(dtype)?)?;
        for block in &self.in_blocks {
            x = block.forward(&x, &emb)?;
        }
        x = resize_spatial(&x, height, width)?;
        for block in &self.out_blocks {
            x = block.forward(&x, &emb)?;
        }
        let x = ops::silu(&self.norm_out.forward(&x)?)?;
        self.conv_out.forward(&x)?.to_dtype(DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn block_plan_interleaves_temporal_blocks_after_every_even_residual() {
        use BlockKind::{Res, Temporal};
        let plan = block_plan();
        assert_eq!(
            plan.len(),
            18,
            "twelve residual blocks plus six temporal blocks"
        );
        let temporal: Vec<usize> = plan
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == Temporal)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(temporal, vec![1, 4, 7, 10, 13, 16]);
        assert_eq!(plan[0], Res);
        assert_eq!(plan[17], Res);
    }

    #[test]
    fn shift_along_zero_fills_the_boundary_in_both_directions() {
        let x = Tensor::from_slice(&[1f32, 2., 3., 4.], (1, 4), &cpu()).unwrap();
        let forward = shift_along(&x, 1, 1).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(forward, vec![vec![2., 3., 4., 0.]]);
        let backward = shift_along(&x, 1, -1).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(backward, vec![vec![0., 1., 2., 3.]]);
        let far = shift_along(&x, 1, 9).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(far, vec![vec![0., 0., 0., 0.]]);
    }

    #[test]
    fn group_norm_is_zero_mean_unit_variance_per_group() {
        let device = cpu();
        let weight = Tensor::ones(GROUPS * 2, DType::F32, &device).unwrap();
        let bias = Tensor::zeros(GROUPS * 2, DType::F32, &device).unwrap();
        let norm = GroupNorm3d { weight, bias };
        let x = Tensor::rand(0f32, 3f32, (1, GROUPS * 2, 2, 2, 2), &device).unwrap();
        let y = norm.forward(&x).unwrap();
        let groups = y.reshape((GROUPS, 2 * 8)).unwrap();
        let mean = groups
            .mean_keepdim(1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let var = groups
            .sqr()
            .unwrap()
            .mean_keepdim(1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (m, v) in mean.iter().zip(var.iter()) {
            assert!(m.abs() < 1e-4, "group mean {m}");
            assert!((v - 1.0).abs() < 1e-3, "group variance {v}");
        }
    }

    #[test]
    fn a_centre_identity_conv3d_passes_its_input_through() {
        let device = cpu();
        // Weight [2, 2, 3, 3, 3] with the identity on the centre tap only.
        let mut weight = vec![0f32; 2 * 2 * 27];
        for channel in 0..2 {
            weight[channel * 2 * 27 + channel * 27 + 13] = 1.0;
        }
        let weight = Tensor::from_vec(weight, (2, 2, 3, 3, 3), &device).unwrap();
        let taps = (0..3)
            .map(|tap| {
                weight
                    .i((.., .., tap, .., ..))
                    .unwrap()
                    .contiguous()
                    .unwrap()
            })
            .collect();
        let conv = Conv3d {
            taps,
            bias: Tensor::zeros(2, DType::F32, &device).unwrap(),
            temporal_padding: 1,
            spatial_padding: 1,
        };
        let x = Tensor::rand(-1f32, 1f32, (1, 2, 3, 4, 5), &device).unwrap();
        let y = conv.forward(&x).unwrap();
        let diff = (y - &x)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "identity conv3d drifted by {diff}");
    }

    #[test]
    fn spatial_resize_keeps_constants_and_time() {
        let device = cpu();
        let x = Tensor::full(0.75f32, (1, 3, 4, 2, 3), &device).unwrap();
        let y = resize_spatial(&x, 4, 6).unwrap();
        assert_eq!(y.dims(), &[1, 3, 4, 4, 6]);
        let max = y
            .affine(1.0, -0.75)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max < 1e-6);
    }

    /// Numerical parity with the ComfyUI node on the released weights.
    ///
    /// Needs the 691 MB bf16 checkpoint, so it is ignored by default:
    /// `MOLD_H3_LATENT_UPSCALER_WEIGHTS=/path/to/...bf16.safetensors cargo test -p mold-ai-candle --lib -- --ignored latent_upscaler`.
    /// Set `MOLD_H3_LATENT_UPSCALER_TEST_DEVICE=cuda` to run it on a GPU.
    #[test]
    #[ignore = "needs the released upscaler weights; set MOLD_H3_LATENT_UPSCALER_WEIGHTS"]
    fn latent_upscaler_matches_the_comfy_node_reference() {
        let Ok(weights) = std::env::var("MOLD_H3_LATENT_UPSCALER_WEIGHTS") else {
            return;
        };
        let device = match std::env::var("MOLD_H3_LATENT_UPSCALER_TEST_DEVICE").as_deref() {
            Ok("cuda") => Device::new_cuda(0).unwrap(),
            _ => Device::Cpu,
        };
        let fixture = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/minimax_h3/latent_upscaler_reference_t4h3w5_s2.safetensors"
        );
        let tensors = candle::safetensors::load(fixture, &device).unwrap();
        let input = &tensors["input"];
        let expected = &tensors["expected"];
        let upscaler = LatentUpscaler::load_file(Path::new(&weights), &device, DType::F32).unwrap();
        let (_, _, _, height, width) = expected.dims5().unwrap();
        let output = upscaler.forward(input, 2.0, height, width).unwrap();
        assert_eq!(output.dims(), expected.dims());
        let diff = (&output - expected).unwrap().abs().unwrap();
        let max_diff = diff.max_all().unwrap().to_scalar::<f32>().unwrap();
        let mean_diff = diff.mean_all().unwrap().to_scalar::<f32>().unwrap();
        let scale = expected
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!(
            "latent upscaler vs node: max abs {max_diff:.3e}, mean relative {:.3e}",
            mean_diff / scale
        );
        assert!(
            max_diff < 2e-3,
            "max abs diff {max_diff} against the node reference"
        );
        assert!(
            mean_diff / scale < 1e-4,
            "mean relative diff {}",
            mean_diff / scale
        );
    }
}
