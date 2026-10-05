"""PyTorch reference for MiniMax-M2.5 attention.

MiniMax normalizes each token's complete Q and K projections before splitting
them into heads. This differs from the per-head QK normalization used by the
current MPK paged-attention task.
"""

import torch


def projection_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """RMS-normalize all heads together for each token.

    ``x`` has shape ``[tokens, heads, head_dim]`` and ``weight`` has shape
    ``[heads * head_dim]``. MiniMax applies this operation before reshaping the
    Q and K projections into heads.
    """
    tokens, heads, head_dim = x.shape
    flat = x.float().reshape(tokens, heads * head_dim)
    variance = flat.square().mean(dim=-1, keepdim=True)
    normalized = flat * torch.rsqrt(variance + eps) * weight.float()
    # Match the checkpoint implementation's round-trip through the projection
    # dtype before the attention calculation continues in float32.
    return normalized.to(x.dtype).reshape(tokens, heads, head_dim).float()


def partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Apply NeoX-style RoPE to the first ``rotary_dim`` channels only."""
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    rotated = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    return torch.cat((x_rot * cos + rotated * sin, x_pass), dim=-1)


def minimax_attention_prefill_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Causal GQA reference for the MiniMax-M2.5 attention contract.

    Inputs are post-projection tensors with shapes ``q=[T,NQ,D]`` and
    ``k/v=[T,NKV,D]``. The result has shape ``[T,NQ,D]`` and dtype float32.
    """
    tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    assert num_q_heads % num_kv_heads == 0
    assert q_norm_weight.numel() == num_q_heads * head_dim
    assert k_norm_weight.numel() == num_kv_heads * head_dim

    qn = projection_rms_norm(q, q_norm_weight, eps)
    kn = projection_rms_norm(k, k_norm_weight, eps)
    qr = partial_rope(qn, cos, sin, rotary_dim)
    kr = partial_rope(kn, cos, sin, rotary_dim)

    group_size = num_q_heads // num_kv_heads
    causal = torch.tril(
        torch.ones(tokens, tokens, dtype=torch.bool, device=q.device)
    )
    output = torch.empty(
        tokens, num_q_heads, head_dim, dtype=torch.float32, device=q.device
    )
    scale = head_dim**-0.5
    for q_head in range(num_q_heads):
        kv_head = q_head // group_size
        scores = (qr[:, q_head] @ kr[:, kv_head].T) * scale
        scores = scores.masked_fill(~causal, float("-inf"))
        output[:, q_head] = torch.softmax(scores, dim=-1) @ v[:, kv_head].float()
    return output
