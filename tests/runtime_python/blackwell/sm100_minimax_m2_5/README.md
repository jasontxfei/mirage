# MiniMax-M2.5 attention

This directory starts the MiniMax-M2.5 Blackwell support tracked by the Fall
2026 MPK roadmap. The first slice is a CPU-testable attention reference using
the public checkpoint contract:

- 48 query heads, 8 KV heads, head dimension 128;
- QK normalization over each token's complete Q or K projection;
- partial RoPE over the first 64 channels, with theta 5,000,000;
- full causal GQA.

Sources: the official
[MiniMax-M2.5 config](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/config.json)
and
[model implementation](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/modeling_minimax_m2.py).

The projection-wide normalization is the important compatibility boundary.
`PersistentKernel.paged_attention_layer` currently accepts 128-element Q/K
normalization weights and normalizes each head independently. MiniMax weights
have 6,144 Q elements and 1,024 K elements and normalize across those complete
projections before the head reshape. Therefore the existing fused path must not
be advertised as MiniMax-compatible without a preprocessing task or a new
cross-head normalization path.

Next step: implement and validate an SM100 QK-normalization + partial-RoPE task
against `pytorch_reference.py`, then feed its output into paged attention with
the existing per-head normalization disabled.
