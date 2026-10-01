- **MiniMax H3 Ref2VA can render in two passes (`refine`).** `GenerateRequest`
  takes an optional `refine: { "scale": 2 }` (CLI: `mold run --refine`) on
  `minimax-h3-ref2va:comfy-pruned-int8-turbo-8step-768p`: the full 8-step
  schedule runs at half the requested size in each axis, the latent is upscaled
  and re-noised at sigma 0.9231, and the schedule's last four steps run at the
  requested (final) size, up to 1920x1088, with the references reused. The
  recipe advertises it as `capabilities.refine`; both final axes must be
  multiples of 64 and `steps` must be 9. Absent keeps the single pass and every
  existing size limit; every other model rejects the field. The output metadata
  records it. The research-only `MOLD_H3_REFINE_PROTO` environment gate is
  gone.
