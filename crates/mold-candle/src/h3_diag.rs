//! THROWAWAY DIAGNOSTIC (never merged): host-to-device weight upload counters
//! for the block-streamed MiniMax H3 denoiser. The reporting half lives in
//! `mold-inference/src/minimax_h3/diag_timing.rs`; read its header for how to
//! enable it and for the log line format.
//!
//! Active only when `MOLD_H3_DIAG_TIMING=1` (read once). When off, [`enabled`]
//! is a cached `false` and no hook calls `Device::synchronize`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use candle::{Device, Result};

static ENABLED: OnceLock<bool> = OnceLock::new();
static H2D_NS: AtomicU64 = AtomicU64::new(0);
static H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static H2D_COPIES: AtomicU64 = AtomicU64::new(0);
static SYNCS: AtomicU64 = AtomicU64::new(0);

/// True when `MOLD_H3_DIAG_TIMING=1`; the environment is read exactly once.
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("MOLD_H3_DIAG_TIMING").is_ok_and(|value| value == "1"))
}

/// Counted `Device::synchronize` (a no-op on CPU). Only called when enabled.
pub fn fence(device: &Device) -> Result<()> {
    SYNCS.fetch_add(1, Ordering::Relaxed);
    device.synchronize()
}

/// Fence and start the clock for one weight upload; `None` when disabled.
pub fn h2d_begin(device: &Device) -> Result<Option<Instant>> {
    if !enabled() {
        return Ok(None);
    }
    fence(device)?;
    Ok(Some(Instant::now()))
}

/// Fence (so the copy has really finished) and account the elapsed time.
pub fn h2d_end(device: &Device, started: Option<Instant>, bytes: u64) -> Result<()> {
    if let Some(started) = started {
        fence(device)?;
        H2D_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        H2D_BYTES.fetch_add(bytes, Ordering::Relaxed);
        H2D_COPIES.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

/// Totals accumulated since the previous call (the counters reset to zero).
#[derive(Clone, Copy, Debug, Default)]
pub struct H2dTotals {
    pub nanos: u64,
    pub bytes: u64,
    pub copies: u64,
    pub fences: u64,
}

pub fn take_totals() -> H2dTotals {
    H2dTotals {
        nanos: H2D_NS.swap(0, Ordering::Relaxed),
        bytes: H2D_BYTES.swap(0, Ordering::Relaxed),
        copies: H2D_COPIES.swap(0, Ordering::Relaxed),
        fences: SYNCS.swap(0, Ordering::Relaxed),
    }
}

/// Fenced per-op spans inside one block forward. Slots: 0 qkv projection,
/// 1 q/k/v reshape + q/k norm + rotary, 2 attention kernel, 3 output
/// projection, 4 fc1, 5 gate activation, 6 fc2. Weight uploads that happen
/// inside a span are subtracted from it (they are reported as `h2d_ms`).
pub const SPAN_COUNT: usize = 14;
/// Slots 7..=11 nest inside the linear spans above: 7 Hadamard rotation,
/// 8 row-wise quantize, 9 cuBLASLt INT8 GEMM, 10 dequantize, 11 Turbo LoRA
/// delta, 12 device allocations of the native op, 13 their release. Only slots 0..=6 are mutually exclusive.
static SPAN_NS: [AtomicU64; SPAN_COUNT] = [const { AtomicU64::new(0) }; SPAN_COUNT];

pub fn span_begin(device: &Device) -> Result<Option<(Instant, u64)>> {
    if !enabled() {
        return Ok(None);
    }
    fence(device)?;
    Ok(Some((Instant::now(), H2D_NS.load(Ordering::Relaxed))))
}

pub fn span_end(device: &Device, started: Option<(Instant, u64)>, slot: usize) -> Result<()> {
    if let Some((at, h2d_before)) = started {
        fence(device)?;
        let h2d = H2D_NS.load(Ordering::Relaxed).saturating_sub(h2d_before);
        let ns = (at.elapsed().as_nanos() as u64).saturating_sub(h2d);
        SPAN_NS[slot].fetch_add(ns, Ordering::Relaxed);
    }
    Ok(())
}

pub fn take_spans() -> [u64; SPAN_COUNT] {
    let mut out = [0u64; SPAN_COUNT];
    for (slot, value) in SPAN_NS.iter().enumerate() {
        out[slot] = value.swap(0, Ordering::Relaxed);
    }
    out
}

#[cfg(feature = "cuda")]
pub fn span_begin_cuda(device: &candle::CudaDevice) -> Result<Option<(Instant, u64)>> {
    span_begin(&Device::Cuda(device.clone()))
}

#[cfg(feature = "cuda")]
pub fn span_end_cuda(
    device: &candle::CudaDevice,
    started: Option<(Instant, u64)>,
    slot: usize,
) -> Result<()> {
    span_end(&Device::Cuda(device.clone()), started, slot)
}
