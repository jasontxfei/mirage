"""SM100 test-mode check for MiniMax projection-wide Q/K preprocessing.

Run on a Blackwell GPU with:

    python test_minimax_qk_norm_rope_testmode.py
"""

import os

import pytest
import torch

# Keep the reference-only tests runnable on development machines that cannot
# load Mirage's CUDA extension.  Device validation still runs normally on
# Blackwell hosts.
if not torch.cuda.is_available():
    pytest.skip("requires a CUDA GPU", allow_module_level=True)

major, _ = torch.cuda.get_device_capability()
if major < 10:
    pytest.skip("requires an SM100-or-newer GPU", allow_module_level=True)

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from pytorch_reference import partial_rope, projection_rms_norm


NUM_Q_HEADS = 48
NUM_KV_HEADS = 8
HEAD_DIM = 128
ROTARY_DIM = 64
Q_HEADS_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
QKV_WIDTH = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM


def _cos_sin(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 5_000_000.0
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, ROTARY_DIM, 2, device="cuda") / ROTARY_DIM)
    )
    positions = torch.arange(tokens, device="cuda", dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def _pack_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pack [Q group, K head, V head] for every KV head."""
    groups = []
    for group in range(NUM_KV_HEADS):
        q_start = group * Q_HEADS_PER_KV
        q_group = q[:, q_start : q_start + Q_HEADS_PER_KV].flatten(1)
        groups.append(
            torch.cat((q_group, k[:, group], v[:, group]), dim=-1)
        )
    return torch.cat(groups, dim=-1).contiguous()


def test_minimax_qk_norm_rope_testmode():
    tokens = 4
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(
        tokens,
        NUM_Q_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    k = torch.randn(
        tokens,
        NUM_KV_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    v = torch.randn(
        tokens,
        NUM_KV_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    q_weight = torch.randn(
        NUM_Q_HEADS * HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    k_weight = torch.randn(
        NUM_KV_HEADS * HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    cos, sin = _cos_sin(tokens)
    packed_qkv = _pack_qkv(q, k, v)
    output = torch.empty_like(packed_qkv)
    assert packed_qkv.shape == (tokens, QKV_WIDTH)

    q_ref = partial_rope(
        projection_rms_norm(q, q_weight), cos, sin, ROTARY_DIM
    ).to(torch.bfloat16)
    k_ref = partial_rope(
        projection_rms_norm(k, k_weight), cos, sin, ROTARY_DIM
    ).to(torch.bfloat16)
    expected = _pack_qkv(q_ref, k_ref, v)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = tokens
    params["max_num_batched_requests"] = tokens
    kernel = PersistentKernel(**params)

    qkv_dt = kernel.attach_input(packed_qkv, name="minimax_qkv")
    q_weight_dt = kernel.attach_input(q_weight, name="minimax_q_norm")
    k_weight_dt = kernel.attach_input(k_weight, name="minimax_k_norm")
    cos_dt = kernel.attach_input(cos, name="minimax_cos")
    sin_dt = kernel.attach_input(sin, name="minimax_sin")
    output_dt = kernel.attach_input(output, name="minimax_output")
    kernel.minimax_qk_norm_rope_layer(
        qkv=qkv_dt,
        q_norm=q_weight_dt,
        k_norm=k_weight_dt,
        cos=cos_dt,
        sin=sin_dt,
        output=output_dt,
        grid_dim=(tokens, 1, 1),
        block_dim=(128, 1, 1),
    )

    try:
        kernel.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
        kernel()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    finally:
        kernel.finalize()


if __name__ == "__main__":
    test_minimax_qk_norm_rope_testmode()
