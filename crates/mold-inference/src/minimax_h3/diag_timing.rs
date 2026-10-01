//! THROWAWAY DIAGNOSTIC (never merged): where does a block-streamed MiniMax H3
//! forward, and the VAE-decode -> H.264 sink, spend its time?
//!
//! Enable with `MOLD_H3_DIAG_TIMING=1` (read once) and make sure the target is
//! visible, e.g. `RUST_LOG=mold::minimax_h3::diag=info`. When the variable is
//! unset every hook is one cached boolean test: no `Device::synchronize`, no
//! clock reads, no allocation, no behaviour change. When set, the hooks add
//! device fences (see below), so the measured wall time is slightly LONGER
//! than an unfenced run; every line reports `fences` and `idle_fence_us` so
//! the direct fence cost can be subtracted.
//!
//! # Forward line (one per `H3BlockStreamedDenoiser::denoise`)
//!
//! ```text
//! H3 diag forward <k>/<n> rows=<r> host_stage_ms=<a> h2d_ms=<b> gpu_ms=<c>
//!   other_ms=<residual> total_ms=<t> bytes_read=<bytes> blocks=<n> ...
//! ```
//!
//! * `k` is the process-wide forward counter; `n` is `?` unless
//!   `MOLD_H3_DIAG_FORWARDS_TOTAL` is set.
//! * `rows` is the packed sequence length (`H3FrozenPackedLayout::seq_len`).
//! * `host_stage_ms` (a): wall time of `H3PrivateComfyBlockLoader::load_block`
//!   (file read + second copy into `Tensor::from_raw_buffer`, plus the few
//!   small dense per-block tensors that `new_comfy_int8` uploads and the Turbo
//!   delta lookup), fenced before and after.
//! * `h2d_ms` (b): sum over the block's INT8 linears of the packed-weight and
//!   scale `to_device` (`ComfyInt8ConvRotLinear::stage_for_native`), fenced
//!   before and after each copy so queued kernels are not billed to it.
//! * `gpu_ms` (c): fenced `forward_block` time minus (b). It is therefore GPU
//!   execution of the activation quantize / INT8 GEMM / dequantize / attention
//!   / norm kernels PLUS the host launch overhead of that block.
//! * `other_ms`: `total_ms - a - b - c` (begin_step, finish_step, projections,
//!   output heads, checkpoints, the fences around the forward itself).
//! * `bytes_read`: checkpoint weight bytes read from disk this forward.
//! * `cpu_s`/`cpu_user_s`/`cpu_sys_s`/`cpu_cores`: process CPU time
//!   (`/proc/self/stat` utime+stime) consumed during the forward, and that
//!   divided by wall time; `proc_rchar`, `proc_read_bytes` (deltas) and
//!   `proc_read_bytes_total` come from `/proc/self/io` (Linux only, else `na`).
//!
//! # Sink lines (VAE decode -> openh264)
//!
//! ```text
//! H3 diag sink chunk <i> frames=<n> decode_ms=<a> d2h_ms=<b> interleave_ms=<c>
//!   yuv_ms=<d> encode_ms=<e> pack_ms=<p> other_ms=<o>
//! H3 diag sink total chunks=<n> frames=<n> decode_ms=... finish_ms=<f> ...
//! ```
//!
//! * `decode_ms`: GPU time between the end of the previous sink write (or the
//!   start of `decode_video`) and the moment the chunk arrives AFTER a device
//!   fence, i.e. VAE decode plus the cheap blend/denormalize ops.
//! * `d2h_ms`: `to_device(Cpu)` + `flatten_all` + `to_vec1` of the F32 chunk.
//! * `interleave_ms`: the scalar planar -> interleaved u8 loop plus
//!   `RgbImage::from_raw`, summed over the chunk's frames.
//! * `yuv_ms`: `YUVBuffer::from_rgb_source`; `encode_ms`: the openh264
//!   `encode` call; `pack_ms`: bitstream copy + NAL splitting in
//!   `Mp4StreamEncoder::push`; all summed over the chunk's frames.
//! * `other_ms`: the rest of the write (sink bookkeeping, checkpoints,
//!   first-frame clone).
//! * `finish_ms` (in the total line) is `H3VideoEncodeSink::finish`:
//!   thumbnail PNG (`thumb_ms`) + MP4 mux (`mux_ms`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use candle_core::{Device, Tensor};

const TARGET: &str = "mold::minimax_h3::diag";

/// True when `MOLD_H3_DIAG_TIMING=1`. Delegates to the one cached read in
/// `mold-candle` so the candle-side and inference-side hooks always agree.
pub(crate) fn enabled() -> bool {
    mold_candle::h3_diag::enabled()
}

/// Start a stopwatch; `None` (no clock read) when the diagnostic is off.
pub(crate) fn tick() -> Option<Instant> {
    enabled().then(Instant::now)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn ms(nanos: u64) -> f64 {
    nanos as f64 / 1.0e6
}

fn since(started: Instant) -> u64 {
    started.elapsed().as_nanos() as u64
}

fn fence(device: &Device) {
    // A failed synchronize will surface on the next real operation.
    let _ = mold_candle::h3_diag::fence(device);
}

// ---------------------------------------------------------------- /proc ----

#[derive(Clone, Copy, Default)]
struct ProcSample {
    user_s: f64,
    sys_s: f64,
    rchar: u64,
    read_bytes: u64,
    available: bool,
}

#[cfg(target_os = "linux")]
fn proc_sample() -> ProcSample {
    let mut sample = ProcSample::default();
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return sample;
    };
    // comm may contain spaces; the numeric fields follow the last ')'.
    let Some(rest) = stat.rsplit_once(')').map(|(_, rest)| rest) else {
        return sample;
    };
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // Field 14 (utime) is index 11 after the state field (field 3 -> index 0).
    let (Some(utime), Some(stime)) = (
        fields.get(11).and_then(|v| v.parse::<u64>().ok()),
        fields.get(12).and_then(|v| v.parse::<u64>().ok()),
    ) else {
        return sample;
    };
    // SAFETY: sysconf has no preconditions.
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as f64;
    sample.user_s = utime as f64 / ticks;
    sample.sys_s = stime as f64 / ticks;
    sample.available = true;
    if let Ok(io) = std::fs::read_to_string("/proc/self/io") {
        for line in io.lines() {
            if let Some(value) = line.strip_prefix("rchar:") {
                sample.rchar = value.trim().parse().unwrap_or(0);
            } else if let Some(value) = line.strip_prefix("read_bytes:") {
                sample.read_bytes = value.trim().parse().unwrap_or(0);
            }
        }
    }
    sample
}

#[cfg(not(target_os = "linux"))]
fn proc_sample() -> ProcSample {
    ProcSample::default()
}

// ------------------------------------------------------------- forward ----

struct Forward {
    index: u64,
    rows: usize,
    device: Device,
    started: Instant,
    host_stage_ns: u64,
    block_ns: u64,
    blocks: u64,
    bytes_read: u64,
    proc_start: ProcSample,
}

static FORWARD: Mutex<Option<Forward>> = Mutex::new(None);
static FORWARD_COUNTER: AtomicU64 = AtomicU64::new(0);
static IDLE_FENCE_NS: OnceLock<u64> = OnceLock::new();

fn idle_fence_ns(device: &Device) -> u64 {
    *IDLE_FENCE_NS.get_or_init(|| {
        fence(device);
        const ROUNDS: u64 = 32;
        let started = Instant::now();
        for _ in 0..ROUNDS {
            fence(device);
        }
        since(started) / ROUNDS
    })
}

/// Scope of one `denoise` call. Inert (and free) when the diagnostic is off.
pub(crate) struct ForwardDiag {
    active: bool,
}

impl ForwardDiag {
    pub(crate) fn begin(device: &Device, rows: usize) -> Self {
        if !enabled() {
            return Self { active: false };
        }
        idle_fence_ns(device);
        fence(device);
        // Discard counters left over from anything outside a forward.
        let _ = mold_candle::h3_diag::take_totals();
        let index = FORWARD_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        *lock(&FORWARD) = Some(Forward {
            index,
            rows,
            device: device.clone(),
            started: Instant::now(),
            host_stage_ns: 0,
            block_ns: 0,
            blocks: 0,
            bytes_read: 0,
            proc_start: proc_sample(),
        });
        Self { active: true }
    }

    /// Fence, then log the single per-forward line.
    pub(crate) fn finish(mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(forward) = lock(&FORWARD).take() else {
            return;
        };
        fence(&forward.device);
        let total_ns = since(forward.started);
        let proc_end = proc_sample();
        let h2d = mold_candle::h3_diag::take_totals();
        let spans = mold_candle::h3_diag::take_spans();
        let gpu_ns = forward.block_ns.saturating_sub(h2d.nanos);
        let span_sum: u64 = spans[..7].iter().sum();
        let rest_ns = gpu_ns.saturating_sub(span_sum);
        let other_ns = total_ns
            .saturating_sub(forward.host_stage_ns)
            .saturating_sub(h2d.nanos)
            .saturating_sub(gpu_ns);
        let total_label = std::env::var("MOLD_H3_DIAG_FORWARDS_TOTAL")
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "?".to_string());
        let idle_us = IDLE_FENCE_NS.get().copied().unwrap_or(0) as f64 / 1.0e3;
        let wall_s = total_ns as f64 / 1.0e9;
        let proc = if forward.proc_start.available && proc_end.available {
            let user = proc_end.user_s - forward.proc_start.user_s;
            let sys = proc_end.sys_s - forward.proc_start.sys_s;
            format!(
                "cpu_s={:.2} cpu_user_s={:.2} cpu_sys_s={:.2} cpu_cores={:.2} \
                 proc_rchar={} proc_read_bytes={} proc_read_bytes_total={}",
                user + sys,
                user,
                sys,
                (user + sys) / wall_s.max(1e-9),
                proc_end.rchar.saturating_sub(forward.proc_start.rchar),
                proc_end
                    .read_bytes
                    .saturating_sub(forward.proc_start.read_bytes),
                proc_end.read_bytes,
            )
        } else {
            "cpu_s=na cpu_user_s=na cpu_sys_s=na cpu_cores=na proc_rchar=na \
             proc_read_bytes=na proc_read_bytes_total=na"
                .to_string()
        };
        tracing::info!(
            target: "mold::minimax_h3::diag",
            "H3 diag spans {}/{} rows={} qkv_ms={:.1} prep_ms={:.1} attn_ms={:.1} out_ms={:.1} \
             fc1_ms={:.1} act_ms={:.1} fc2_ms={:.1} rest_ms={:.1} gpu_ms={:.1} \
             inner_rotate_ms={:.1} inner_quant_ms={:.1} inner_gemm_ms={:.1} inner_dequant_ms={:.1} \
             inner_turbo_ms={:.1}",
            forward.index,
            total_label,
            forward.rows,
            ms(spans[0]),
            ms(spans[1]),
            ms(spans[2]),
            ms(spans[3]),
            ms(spans[4]),
            ms(spans[5]),
            ms(spans[6]),
            ms(rest_ns),
            ms(gpu_ns),
            ms(spans[7]),
            ms(spans[8]),
            ms(spans[9]),
            ms(spans[10]),
            ms(spans[11]),
        );
        tracing::info!(
            target: "mold::minimax_h3::diag",
            "H3 diag forward {}/{} rows={} host_stage_ms={:.1} h2d_ms={:.1} gpu_ms={:.1} \
             other_ms={:.1} total_ms={:.1} bytes_read={} blocks={} h2d_bytes={} h2d_copies={} \
             fences={} idle_fence_us={:.1} {}",
            forward.index,
            total_label,
            forward.rows,
            ms(forward.host_stage_ns),
            ms(h2d.nanos),
            ms(gpu_ns),
            ms(other_ns),
            ms(total_ns),
            forward.bytes_read,
            forward.blocks,
            h2d.bytes,
            h2d.copies,
            h2d.fences,
            idle_us,
            proc,
        );
    }
}

impl Drop for ForwardDiag {
    fn drop(&mut self) {
        if self.active {
            // Errored or cancelled forward: drop the partial state silently.
            *lock(&FORWARD) = None;
            let _ = mold_candle::h3_diag::take_totals();
        }
    }
}

/// Opaque stopwatch for one `load_block`.
pub(crate) struct HostStage {
    started: Instant,
    bytes_read_before: u64,
}

/// Call before `load_block`. Fences the forward's device so queued GPU work is
/// not billed to host staging.
pub(crate) fn host_stage_begin(bytes_read: u64) -> Option<HostStage> {
    if !enabled() {
        return None;
    }
    let device = lock(&FORWARD)
        .as_ref()
        .map(|forward| forward.device.clone())?;
    fence(&device);
    Some(HostStage {
        started: Instant::now(),
        bytes_read_before: bytes_read,
    })
}

/// Call after `load_block` with the loader's cumulative `bytes_read`.
pub(crate) fn host_stage_end(stage: Option<HostStage>, bytes_read: u64) {
    let Some(stage) = stage else { return };
    let mut guard = lock(&FORWARD);
    let Some(forward) = guard.as_mut() else {
        return;
    };
    fence(&forward.device);
    forward.host_stage_ns += since(stage.started);
    forward.bytes_read += bytes_read.saturating_sub(stage.bytes_read_before);
}

/// Call immediately before one block's `forward_block`: fence, start clock.
pub(crate) fn block_begin(device: &Device) -> Option<Instant> {
    if !enabled() || lock(&FORWARD).is_none() {
        return None;
    }
    fence(device);
    Some(Instant::now())
}

/// Call immediately after `forward_block`: fence, account the block.
pub(crate) fn block_end(device: &Device, started: Option<Instant>) {
    let Some(started) = started else { return };
    fence(device);
    let elapsed = since(started);
    if let Some(forward) = lock(&FORWARD).as_mut() {
        forward.block_ns += elapsed;
        forward.blocks += 1;
    }
}

// ---------------------------------------------------------------- sink ----

#[derive(Clone, Copy, Default)]
struct SinkParts {
    frames: u64,
    decode_ns: u64,
    d2h_ns: u64,
    interleave_ns: u64,
    yuv_ns: u64,
    encode_ns: u64,
    pack_ns: u64,
    other_ns: u64,
}

impl SinkParts {
    fn add(&mut self, other: &SinkParts) {
        self.frames += other.frames;
        self.decode_ns += other.decode_ns;
        self.d2h_ns += other.d2h_ns;
        self.interleave_ns += other.interleave_ns;
        self.yuv_ns += other.yuv_ns;
        self.encode_ns += other.encode_ns;
        self.pack_ns += other.pack_ns;
        self.other_ns += other.other_ns;
    }

    fn fields(&self) -> String {
        format!(
            "decode_ms={:.1} d2h_ms={:.1} interleave_ms={:.1} yuv_ms={:.1} encode_ms={:.1} \
             pack_ms={:.1} other_ms={:.1}",
            ms(self.decode_ns),
            ms(self.d2h_ns),
            ms(self.interleave_ns),
            ms(self.yuv_ns),
            ms(self.encode_ns),
            ms(self.pack_ns),
            ms(self.other_ns),
        )
    }
}

struct Sink {
    last_end: Instant,
    chunk_started: Option<Instant>,
    chunks: u64,
    chunk: SinkParts,
    total: SinkParts,
    thumb_ns: u64,
    mux_ns: u64,
}

static SINK: Mutex<Option<Sink>> = Mutex::new(None);

fn with_sink(update: impl FnOnce(&mut Sink)) {
    if let Some(sink) = lock(&SINK).as_mut() {
        update(sink);
    }
}

/// Call at the start of `decode_video`: resets the job totals.
pub(crate) fn sink_begin() {
    if !enabled() {
        return;
    }
    *lock(&SINK) = Some(Sink {
        last_end: Instant::now(),
        chunk_started: None,
        chunks: 0,
        chunk: SinkParts::default(),
        total: SinkParts::default(),
        thumb_ns: 0,
        mux_ns: 0,
    });
}

/// Start of `DecodeSink::write`: fence the chunk's device so the preceding
/// asynchronous decode is fully accounted as `decode_ms`.
pub(crate) fn sink_chunk_begin(frames: &Tensor) -> bool {
    if !enabled() || lock(&SINK).is_none() {
        return false;
    }
    fence(frames.device());
    with_sink(|sink| {
        let now = Instant::now();
        sink.chunk = SinkParts {
            decode_ns: now.duration_since(sink.last_end).as_nanos() as u64,
            ..SinkParts::default()
        };
        sink.chunk_started = Some(now);
    });
    true
}

pub(crate) fn sink_d2h(started: Option<Instant>) {
    if let Some(started) = started {
        let elapsed = since(started);
        with_sink(|sink| sink.chunk.d2h_ns += elapsed);
    }
}

pub(crate) fn sink_interleave(started: Option<Instant>) {
    if let Some(started) = started {
        let elapsed = since(started);
        with_sink(|sink| sink.chunk.interleave_ns += elapsed);
    }
}

pub(crate) fn sink_yuv(started: Option<Instant>) {
    if let Some(started) = started {
        let elapsed = since(started);
        with_sink(|sink| sink.chunk.yuv_ns += elapsed);
    }
}

pub(crate) fn sink_encode(started: Option<Instant>) {
    if let Some(started) = started {
        let elapsed = since(started);
        with_sink(|sink| sink.chunk.encode_ns += elapsed);
    }
}

pub(crate) fn sink_pack(started: Option<Instant>) {
    if let Some(started) = started {
        let elapsed = since(started);
        with_sink(|sink| sink.chunk.pack_ns += elapsed);
    }
}

/// End of `DecodeSink::write`: log the chunk line and fold it into the totals.
pub(crate) fn sink_chunk_end(active: bool, frames: usize) {
    if !active {
        return;
    }
    let mut guard = lock(&SINK);
    let Some(sink) = guard.as_mut() else { return };
    let Some(started) = sink.chunk_started.take() else {
        return;
    };
    let wall_ns = since(started);
    sink.chunk.frames = frames as u64;
    sink.chunk.other_ns = wall_ns
        .saturating_sub(sink.chunk.d2h_ns)
        .saturating_sub(sink.chunk.interleave_ns)
        .saturating_sub(sink.chunk.yuv_ns)
        .saturating_sub(sink.chunk.encode_ns)
        .saturating_sub(sink.chunk.pack_ns);
    tracing::info!(
        target: "mold::minimax_h3::diag",
        "H3 diag sink chunk {} frames={} {}",
        sink.chunks,
        frames,
        sink.chunk.fields(),
    );
    let chunk = sink.chunk;
    sink.total.add(&chunk);
    sink.chunks += 1;
    sink.last_end = Instant::now();
}

/// Time the thumbnail PNG inside `H3VideoEncodeSink::finish`.
pub(crate) fn sink_thumbnail(started: Option<Instant>) {
    if let Some(started) = started {
        let elapsed = since(started);
        with_sink(|sink| sink.thumb_ns += elapsed);
    }
}

/// End of `H3VideoEncodeSink::finish`: log the job total line.
pub(crate) fn sink_finish(started: Option<Instant>) {
    let Some(started) = started else { return };
    let finish_ns = since(started);
    let Some(sink) = lock(&SINK).take() else {
        return;
    };
    let mux_ns = finish_ns.saturating_sub(sink.thumb_ns);
    tracing::info!(
        target: "mold::minimax_h3::diag",
        "H3 diag sink total chunks={} frames={} {} finish_ms={:.1} thumb_ms={:.1} mux_ms={:.1}",
        sink.chunks,
        sink.total.frames,
        sink.total.fields(),
        ms(finish_ns),
        ms(sink.thumb_ns),
        ms(mux_ns),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_hooks_do_nothing() {
        // The test environment does not set MOLD_H3_DIAG_TIMING.
        if enabled() {
            return;
        }
        assert!(tick().is_none());
        let diag = ForwardDiag::begin(&Device::Cpu, 7);
        assert!(!diag.active);
        assert!(host_stage_begin(0).is_none());
        assert!(block_begin(&Device::Cpu).is_none());
        diag.finish();
        let frames = Tensor::zeros((1, 3, 1, 1, 1), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(!sink_chunk_begin(&frames));
        sink_chunk_end(false, 1);
        sink_finish(None);
        assert_eq!(mold_candle::h3_diag::take_totals().fences, 0);
    }
}
