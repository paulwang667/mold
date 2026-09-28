- **Two more MiniMax H3 Ref2VA Turbo tiers.**
  `minimax-h3-ref2va:comfy-pruned-int8-turbo-8step-768p` pulls the Ref2VA
  compact stack plus lightx2v's 8-step 768p Ref2V adapter (1,956,193,000
  bytes, pinned at lightx2v revision `0eebcc7e…`, the first that publishes
  it) and renders 9 terminal-inclusive steps with Euler at video shift 12 /
  audio shift 3, as its publisher's release note recommends.
  `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step-s050` is a **draft** tier:
  the existing Ref2VA 4-step adapter applied at strength 0.5 (every LoRA delta
  halved), because at full strength that adapter systematically places the
  subject left of centre; it shares the 4-step tag's adapter file, so a host
  that already has `-turbo-4step` downloads nothing new. The draft strength is
  part of the tier's identity, so a full-strength render is never reused for
  it ([#814](https://github.com/utensils/mold/issues/814)).
- **MiniMax H3 streamed pipelines honour the tier's video shift.** The
  block-streamed FL2VA and Ref2VA pipeline wrappers now forward the frozen
  tier's declared video shift instead of falling back to 12, so a tier with
  another shift can never sample a different sigma grid than its authority
  names ([#814](https://github.com/utensils/mold/issues/814)).
