- **MiniMax H3 Turbo LoRA strength is a per-request control.** `GenerateRequest`
  takes an optional `turbo_lora_strength` in `(0, 1]` (CLI: `mold run
  --turbo-strength`) that replaces the published strength (`1.0`) for that
  render, so a client can sweep it instead of picking a fixed tier. Absent keeps
  the published strength, and `1.0` freezes the identical
  authority; any other value is folded into the frozen adapter identity, so a
  render at one strength is never reused for another. Rejected, not ignored,
  on non-Turbo H3 tags and on every other family. Distinct from `strength`,
  the denoise strength H3 fixes at 1. The strength is a request field, not a
  tier: the never-released `-turbo-4step-s050` draft tag (the Ref2VA 4-step
  adapter at half strength) is gone, and `turbo_lora_strength: 0.5` on
  `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step` is the recommended way to
  get its behavior (it corrects the full-strength v0.1 adapter's left-of-centre
  subject placement).
