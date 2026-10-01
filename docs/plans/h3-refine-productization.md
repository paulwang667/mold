# MiniMax H3 `refine` — hires-fix second pass, productization plan

Status: plan, 2026-10-01. Branch `aiva/h3-refine`, based on the research
prototype `aiva/h3-refine-proto` (env-gated; measurements in "Evidence").
Phases 1 and 2 landed together on this branch (2026-10-01): the contract, door
rules, canvas-rule split, capability advertisement, plan-driven pipeline, ledger
and two-pass provenance exist and the environment gate is gone. Phase 3a
removed the memory waivers in `private_server.rs`: strict admission applies to a
refine request exactly as to any other. The open bitrate question and the rest
of phases 3 to 6 remain.

Fork-only: this does not target upstream `utensils/mold`, so the upstream
reviewed-IDs-only / qualification-record rules bind only where they already bind
this fork's H3 route (they do; see Constraints).

## What it is

An opt-in request block that renders a Ref2VA clip in two passes:

1. **Pass 1** — the full distilled schedule (8 forwards) on a canvas half the
   final size in each axis, to a clean latent.
2. **Upscale** — bilinear 2x spatial interpolation of the unpatchified video
   latent, time untouched, F32.
3. **Re-noise** — `x = σ·ε + (1-σ)·x0` at the schedule's grid index 4
   (σ_video = 0.9231, σ_audio = 0.75) with fresh noise from separate seeded
   streams; the audio latent is re-noised at its matching sigma.
4. **Pass 2** — the last 4 forwards of the same frozen schedule (Euler tail) on
   the final canvas, the references and text conditioning reused, then decode.

The request's `width`/`height` are the FINAL canvas (the LTX-2 convention), so
output size, metadata, gallery and `validate_output` keep meaning what they mean
today.

## Decisions (user, 2026-10-01)

| Decision | Choice |
|---|---|
| Tiers | `minimax-h3-ref2va:comfy-pruned-int8-turbo-8step-768p` only |
| Final canvas ceiling (phase 1) | ≤ 2,088,960 px (1920×1088); 2560×1408 stays prototype-only |
| Where it lives | this fork only |
| Default | off; the request enables it explicitly |
| Fixed in phase 1 (not request fields) | scale = 2, start = grid index 4, pass-2 LoRA strength = the request's `turbo_lora_strength` |

Rejected for phase 1, with the reason: scale ≠ 2 (prototype interpolates integer
scales only; 1.5x needs code and a new evidence round); `start` as a field (the
sweep showed index 5 leaves pass-1 artefacts, index 3 is slower and softer — a
free knob invites the bad setting); `lora2` as a field (pass 2 at strength 1.0
pulled the subject back to the left bias).

## Three lines (owner / readers / what changes together)

- **Owner.** The fact "this request renders in two passes, at this geometry" is
  owned by ONE value, `RefinePlan { scale, start_index }`, derived in
  `mold_core::minimax_h3` from `GenerateRequest.refine` and the final canvas
  (`refine_pass1_canvas`, `validate_refine`). Nothing else re-derives pass-1
  dims, the start index, or the forward counts.
- **Readers.** `validate_request_contract_with_authorities` (door), admission
  and the prepared-request/target-budget identities
  (`mold-inference::h3_factory`, `private_server`), the Ref2VA pipeline
  (`pipeline/ref2va.rs` + `refine.rs`), the phase ledger
  (`private_fl2va_runtime.rs`), `OutputMetadata`/provenance, the CLI, the
  generation profile / capabilities, and aiva's `MoldVideoBackend`.
- **Changes together.** Request type + metadata + docs/generated files + TS
  types, identity hashing (version-gated so a request without `refine` hashes
  byte-identically), the runtime code identity and its private-UAT captures,
  the release-contract script anchors, aiva's artifact fingerprint and cost
  model.

## Contract

```json
{ "model": "minimax-h3-ref2va:comfy-pruned-int8-turbo-8step-768p",
  "width": 1920, "height": 1088, "steps": 9, "frames": 107,
  "turbo_lora_strength": 0.5,
  "refine": { "scale": 2 } }
```

- `refine` is `Option<RefineRequest { scale: u32 }>`, `skip_serializing_if =
  is_none`, mirrored in `OutputMetadata`. Phase 1 accepts only `scale == 2`.
- Door rules (codes in `MINIMAX_H3_REFINE_*`): Ref2VA Turbo 8-step 768p only;
  both final axes multiples of 64 (so pass 1 is on the 32 grid); final area ≤
  `REFINE_MAX_PIXELS`; pass-1 canvas must itself satisfy the existing admitted
  canvas rule; Euler-tail tiers only; rejected, never ignored, on every other
  model and family (the `turbo_lora_strength` precedent).
- The existing canvas ceilings (`COMPACT_MAX_PIXELS`, `MAX_PIXELS`) stay as they
  are for non-refine requests. A refine request is checked against the refine
  rule INSTEAD of the compact ceiling for its FINAL canvas and against the
  compact rule for its PASS-1 canvas. The prototype's `OnceLock` env reader and
  `uncap` gate are deleted.

## Runtime

- `pipeline/refine.rs` (from `refine_proto.rs`): `RefinePlan`, the bilinear
  matmul upsample, `renoise_at_sigma`, the two re-noise streams (seeds XOR
  distinct constants; pass-1 draw order untouched), the log lines. No env.
- `execute_staged` takes `Option<RefinePlan>` from the prepared request, never
  from the environment.
- Ledger: `expected_denoise_forwards = n + (n - start)` for a refine request,
  one resident transformer across both passes.
- Provenance records both passes (final dims, pass-1 dims, forwards per pass,
  start index, sigma) — today it records pass 1 only.
- Bounded thumbnail: the shrink-before-encode fix stays; the 4 MiB bound and the
  8 MiB host bound stay (they feed the runtime qualification).
- MP4 bitrate: the sink is fixed at 10 Mbps openh264; decide in phase 2 whether
  to scale it with area (measure, do not guess).

## Admission, budget, identity (the risky part)

- Admission is priced at the FINAL canvas, the larger pass; pass 1 is smaller.
  The conditioning (text, vision, reference rows) is canvas-independent.
- Add `RefinePlan` and both passes' row counts to the prepared-request input;
  the hash gains the new bytes ONLY when `refine` is present. A test pins the
  no-refine identity as byte-identical to before.
- The prototype waived the extrapolated memory refusals under `uncap`; the
  waivers are gone (phase 3a). A 2560×1408 refine render on a 46 GB L20 ran with
  zero waiver log lines, i.e. the strict checks already accepted the whole
  phase-1 range (final canvas <= 1920×1088), so the waivers were never
  exercised. Neither memory gate takes a refine argument any more; a refine
  request whose predicted peak exceeds the sample is refused with the same typed
  headroom shortfall as any other request. Phase 3 still measures the PEAK (not
  the free-at-end figure) for 1344×768, 1536×640, 1920×1088 and may re-derive
  `public_ref2va_runtime_bounds_for_shape` for the refine range from those,
  with a margin. Measured so far (free VRAM at the end of pass 2, 46 GB L20):
  1920×1088 ≈ 23 GB used, 2560×1408 ≈ 33 GB used.
- Retained across pass 2: pass-1 latents and the text states (small); add a
  `refine_retained_bytes` term.

## Constraints that bind this work

- Generated files are never hand-edited; CI runs the generators with `--check`.
- A `changelog.d/<slug>.md` fragment; docs: `.claude/rules/minimax-h3.md`,
  `docs/qualification/minimax-h3.md`, the website pages, the CLI skill text.
- `scripts/tests/minimax-h3-private-uat-release-contract.sh` text-anchors the
  Ref2VA phase order (encode_text < encode_visual < park < transformer load <
  transformer drop < VAE reload < VAE drop < echo < mux). The refine splice must
  not add a VAE load or a transformer phase in those impls.
- `PRIVATE_RUNTIME_CODE_IDENTITY_SHA256` hashes every runtime `.rs`; any change
  changes it by design, and the private-UAT runtime qualification must be
  re-captured on a CUDA host.
- `h3` server code does not compile on the Mac (build.rs feature-set check);
  server-side tests run on the CUDA host (`scripts/test-h3-cuda-server.sh`).

## Phases

| # | Content | Exit criterion |
|---|---|---|
| 1 | Contract: `RefineRequest`, `OutputMetadata`, door rules, canvas rule split, generation-profile/capability advertisement, generated files, TS types, CLI flag, delete the env gate | `cargo check` of the touched crates; unit tests for every rule; no-refine requests unchanged |
| 2 | Pipeline: `refine.rs`, plan-driven `execute_staged`, ledger, two-pass provenance, bitrate decision | synthetic-backend tests for gated and closed paths; release-contract script passes unmodified |
| 3 | Admission, budget, identity: prepared-request + target-budget identity (version-gated), row caps, (waivers removed, 3a) measure peaks and re-derive bounds | identity byte-equality test; measured peak table committed to `docs/qualification/minimax-h3.md` |
| 4 | CI + qualification: CUDA-host server tests, runtime-identity re-capture, docs, changelog | CI contracts green; recorded captures |
| 5 | aiva: config, final-canvas selection, artifact fingerprint, cost model, run card, draft tier, meta | aiva four gates; no-refine artifacts unchanged |
| 6 | Validation: multi-seed, second character/scene, identity and temporal stability, A/B vs post-hoc SR, learned upscaler decision | written verdict, kept in memory and qualification doc |

**Phase 3c: host block cache.** Measured on an L20 with the 8-step refine
(50 INT8 blocks, 19.3 GB per forward), each forward spent about 11.1 s in host
staging (`H3ComfyInt8BlockLoader::load_block`: per tensor a zero-filled `Vec`,
a locked seek+read, then `from_raw_buffer` copying the bytes again), 2.2 s on
H2D, and 11.7 s (pass 1) / 37.9 s (pass 2) of GPU compute, all serialized, so
the GPU was about 46% busy in pass 1 and system CPU time was about 14 s per
forward. The same 12 forwards (8 + 4) re-read identical bytes.

- Reads reserve capacity instead of zero-filling, and the raw I8 buffer becomes
  the tensor storage: one host copy per packed tensor.
- On CUDA each loader keeps the four packed linears of every block it has read
  (per loader, so per job; CPU and Metal are unchanged). Forwards after the
  first skip the packed reads; the block's small dense tensors are still read
  through the var builder.
- The device contract is untouched: one staged block, no extra device buffer.
  The host contract is not: the ledger's denoise phase now charges every
  block's `encoded_host_bytes` plus two copies of the largest tensor
  (`denoise_block_host_bytes`), under strict admission with no waiver. The
  budget and attempt identities change by design.
- Output stays bit-identical. The new timings are to be recorded on the CUDA
  host.

## Evidence the prototype produced (single character/scene, seed 20260930 unless noted)

- Sharpness (Laplacian variance) at one output size: +53% (1344×576) and +69%
  (1536×640, vs 3-seed direct mean) over a direct render at LoRA strength 0.5;
  position stays near centre.
- At a common 1920 width, the 1920×1088 refine render has 52% more Laplacian
  variance and more high-frequency energy than the 1536×640 refine render
  stretched to 1920, so the extra detail is real.
- Start index sweep (3/4/5 × 2 seeds): position unchanged (0.527–0.534);
  index 5 shows floating gold specks / ghost trails in both seeds, so its higher
  sharpness figure is artefact; index 4 kept.
- Cost: pass 2 is ~45 s/forward at 1536×640, ~104 s at 1920×1088, ~243 s at
  2560×1408 (compute-bound as the canvas grows).
- A full-size thumbnail PNG of a 2560×1408 frame exceeds the 4 MiB bound and
  failed a finished job; fixed by shrinking above 1344×768 before encoding.

## Open items

- Learned latent upscaler vs bilinear (community 24-channel H3 upscalers exist;
  licences unclear) — phase 6 decision, not phase 1.
- Whether the 2560×1408 tier is ever promoted: needs a flash-attention
  qualification above ~107k rows and a memory bound at that size.
- aiva's choice of final aspect ratio (scene references are 2.39:1; 16:9 final
  canvases change composition).
