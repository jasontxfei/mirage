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

## Reuse boundary

MiniMax support should reuse rather than duplicate these existing components:

- `attention_sm100.cuh` already implements paged GQA, KV-cache handling, and
  partial RoPE. The MiniMax-specific gap is projection-wide Q/K normalization;
  it does not require another attention inner loop.
- `topk_sigmoid_sm100.cuh` already implements 256-expert sigmoid-plus-bias
  routing. Instantiating it with one group, top-8, and a scaling factor of one
  matches the MiniMax routing rule; it does not require a new router.
- The static-megakernel compiler and task ABI are being developed in PR #786.
  Its current example supplies Kimi MoE task bodies, but no GQA task family.
- Issue #780 separately tracks a library-level GQA template for GPT-OSS. Any
  generic GQA-template work should be coordinated there rather than duplicated
  under the MiniMax model work.

## Implemented slice

`minimax_qk_norm_rope_sm100.cuh` implements the missing preprocessing task for
the MiniMax-M2.5 geometry. One task processes one token's KV-head-interleaved
QKV row, reduces Q across 6,144 elements and K across 1,024 elements, applies
the projection-wide weights, performs the bf16 round-trip, rotates the first 64
channels of every Q/K head, and copies V unchanged. The task is registered in
`PersistentKernel.minimax_qk_norm_rope_layer` and has an SM100 test-mode check
against `pytorch_reference.py`.

The test still needs to be compiled and run on Blackwell. This development
machine has no CUDA toolkit or GPU. After device validation, connect the task's
packed output to shared paged GQA with the existing per-head normalization and
RoPE disabled. Port the existing sigmoid router into the static task ABI when
the MiniMax MoE slice begins.
