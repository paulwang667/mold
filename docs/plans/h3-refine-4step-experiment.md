# MiniMax H3 `refine` on the Ref2VA Turbo 4-step tier (EXPERIMENT)

Status: throwaway experiment, 2026-10-01. Branch `aiva/h3-refine-4step-exp`,
based on `aiva/h3-refine` (see [`h3-refine-productization.md`](h3-refine-productization.md)).
**Never merge this branch into `aiva/h3-refine`.** If the experiment is worth
keeping it is re-done there as a reviewed decision (the productization plan
restricts `refine` to the 8-step tier on purpose).

## What it is

`aiva/h3-refine` runs the request-driven two-pass `refine`
(`GenerateRequest.refine = {"scale": 2}`) on the Ref2VA Turbo 8-step 768p tier
only: pass 1 is the full 8 forwards at half canvas, pass 2 the last 4 forwards
(grid index 4, `sigma_video = 0.9231`) at the final canvas.

This branch lets the same feature run on
`minimax-h3-ref2va:comfy-pruned-int8-turbo-4step` (`REF2VA_COMFY_TURBO_4STEP`):
5 grid points, shift 12, `ComfyEuler`, video sigmas `[1, .9730, .9231, .8, 0]`.
Index 2 is the SAME `sigma_video = 0.9231` (audio 0.75) as the 8-step tier's
index 4, so the re-noise point is unchanged:

| tier | `steps` | pass 1 | start index | pass 2 | ledger forwards |
|---|---|---|---|---|---|
| 8-step 768p | 9 | 8 at half canvas | 4 | 4 at final canvas | 12 |
| 4-step (experiment) | 5 | 4 at half canvas | 2 | 2 at final canvas | 6 |

The other 4-step tier (`...-turbo-4step-r21`, rank-21 resize) is NOT enabled.

## What changed

- `mold-core::minimax_h3`: `refine_start_index_for_model(model) -> Option<u32>`
  (8-step -> 4, 4-step -> 2, everything else `None`) is the one tier -> index map.
  `refine_supported_model`, `RefinePlan::for_request` / `for_model_scale`, the
  door (`MINIMAX_H3_REFINE_TIER` and `MINIMAX_H3_REFINE_GRID`: steps == the
  tier's grid points) all read it. `H3_REFINE_START_INDEX` still means the
  8-step tier; `RefinePlan::PUBLISHED` is still the 8-step plan.
- `mold-inference::h3_factory`: the prepared-request validator holds the plan to
  `RefinePlan::for_grid_points_scale(request.grid_points, ..)` (9 points -> 4,
  5 points -> 2). A prepared request carries its grid but NOT the tier tag: its
  `canonical_model` is the base partition (`REF2VA_COMFY`), so a lookup by
  model there refused every real 4-step refine request ("prepared request
  authority is internally inconsistent"). `refine_start_index_for_grid_points`
  is derived from the same tier table as `refine_start_index_for_model`.
- `mold-server::h3_admission`: the prepared shape carries no model, so it checks
  `RefinePlan::is_tier_plan` (any tier's plan); the model-aware comparison
  already happened at the request door.
- CLI `--refine` gate: accepts both tiers (message names both).
- Generation profile: `capabilities.refine` is derived from
  `refine_supported_model` and the tier's own `steps`, so the 4-step recipe now
  advertises it (`steps = 5`) without a code change there.
- Everything else (pipeline, `refine::pass2_forwards`, ledger, provenance)
  already took the start index from the `RefinePlan`; only tests were added.

The 8-step tier is byte-identical: the no-refine and 8-step identity pins pass
unchanged.

## How to run it

Request (final canvas = request size, 64-aligned, at most 1920x1088):

```json
{ "model": "minimax-h3-ref2va:comfy-pruned-int8-turbo-4step",
  "width": 1344, "height": 576, "steps": 5, "frames": 107,
  "refine": { "scale": 2 } }
```

CLI: `mold run minimax-h3-ref2va:comfy-pruned-int8-turbo-4step ... --refine`.

Runtime needs a CUDA host (the `h3` server does not build on the Mac). Expect
the log lines `H3 refine pass 1: ... forwards=4` and
`H3 refine pass 2: ... forwards=2`, `sigma=0.9231..0.0000`. Note the private-UAT
runtime code identity and its captures are NOT re-recorded for this branch.

## Open questions this experiment is for

- Does a 2-forward pass 2 recover enough detail at sigma 0.9231, or is the 4-step
  tier's coarser tail too short for the upscale to rebuild texture?
- The turbo LoRA for the 4-step tier is a different adapter (`v0.1`) than the
  8-step tier's (`v1.0 768p`); its left-bias behaviour at pass 2 is unmeasured.
- Peak memory: the factory/admission memory models are tier-independent, but the
  refine peak was measured on the 8-step tier only.
