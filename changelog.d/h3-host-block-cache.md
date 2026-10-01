- **MiniMax H3 CUDA denoise keeps the packed transformer blocks in host
  memory.** Each forward used to re-read all fifty INT8 blocks (19.3 GB) from
  disk and copy every tensor twice on the host; tensors are now read once with
  a single host copy, and on CUDA the packed blocks stay resident for the rest
  of the job, so forwards after the first skip the file reads. Device memory is
  unchanged (one staged block). The per-attempt host budget now charges the
  whole packed checkpoint during denoise, so a host with less free RAM than
  that is refused at admission instead of being admitted on the old one-block
  figure. Output is bit-identical.
