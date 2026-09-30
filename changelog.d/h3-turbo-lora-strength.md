- **MiniMax H3 Turbo LoRA strength is a per-request control.** `GenerateRequest`
  takes an optional `turbo_lora_strength` in `(0, 1]` (CLI: `mold run
  --turbo-strength`) that replaces the Turbo tier's own strength for that
  render, so a client can sweep it instead of picking a fixed tier. Absent keeps
  the reviewed strength, and a value equal to it freezes the identical
  authority; any other value is folded into the frozen adapter identity, so a
  render at one strength is never reused for another. Rejected, not ignored,
  on non-Turbo H3 tags and on every other family. Distinct from `strength`,
  the denoise strength H3 fixes at 1.
