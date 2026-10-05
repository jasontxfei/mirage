"""CPU checks for the MiniMax-M2.5 attention reference contract."""

import torch

from pytorch_reference import (
    minimax_attention_prefill_ref,
    partial_rope,
    projection_rms_norm,
)


NUM_Q_HEADS = 48
NUM_KV_HEADS = 8
HEAD_DIM = 128
ROTARY_DIM = 64


def _cos_sin(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 5_000_000.0
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)
    )
    angles = torch.outer(torch.arange(tokens, dtype=torch.float32), inv_freq)
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos(), emb.sin()


def test_projection_norm_spans_all_heads():
    x = torch.ones(1, 2, 4)
    x[:, 1] = 3
    out = projection_rms_norm(x, torch.ones(8))

    # One statistic is shared by both heads. Per-head normalization would make
    # both head RMS values equal to one instead.
    global_rms = out.square().mean().sqrt()
    head_rms = out.square().mean(dim=-1).sqrt()
    torch.testing.assert_close(global_rms, torch.tensor(1.0))
    assert head_rms[0, 0] < 1.0 < head_rms[0, 1]


def test_partial_rope_preserves_non_rotary_channels():
    torch.manual_seed(0)
    x = torch.randn(3, 2, HEAD_DIM)
    cos, sin = _cos_sin(3)
    out = partial_rope(x, cos, sin, ROTARY_DIM)
    torch.testing.assert_close(out[..., ROTARY_DIM:], x[..., ROTARY_DIM:])


def test_minimax_m2_5_attention_geometry_and_causality():
    torch.manual_seed(0)
    tokens = 4
    q = torch.randn(tokens, NUM_Q_HEADS, HEAD_DIM)
    k = torch.randn(tokens, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(tokens, NUM_KV_HEADS, HEAD_DIM)
    q_weight = torch.randn(NUM_Q_HEADS * HEAD_DIM)
    k_weight = torch.randn(NUM_KV_HEADS * HEAD_DIM)
    cos, sin = _cos_sin(tokens)

    out = minimax_attention_prefill_ref(
        q, k, v, q_weight, k_weight, cos, sin, ROTARY_DIM
    )
    assert out.shape == (tokens, NUM_Q_HEADS, HEAD_DIM)
    assert torch.isfinite(out).all()

    # At position zero, causal attention has exactly one available KV token.
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    for q_head in range(NUM_Q_HEADS):
        torch.testing.assert_close(out[0, q_head], v[0, q_head // group_size])
