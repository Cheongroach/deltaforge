# SPDX-License-Identifier: Apache-2.0
"""Functional fused recurrent GDN step for CUDA; no in-place cache mutation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(Q, K, V, G, BETA, H0, OUT, H1,
            HK: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
            BK: tl.constexpr, BV: tl.constexpr):
    bh = tl.program_id(0)
    value_tile = tl.program_id(1)
    batch, head = bh // HV, bh % HV
    key_head = head // (HV // HK)
    rk = tl.arange(0, BK)
    rv = value_tile * BV + tl.arange(0, BV)
    q = tl.load(Q + (batch * HK + key_head) * DK + rk, rk < DK, 0).to(tl.float32)
    k = tl.load(K + (batch * HK + key_head) * DK + rk, rk < DK, 0).to(tl.float32)
    v = tl.load(V + bh * DV + rv, rv < DV, 0).to(tl.float32)
    gate = tl.exp(tl.load(G + bh).to(tl.float32))
    beta = tl.load(BETA + bh).to(tl.float32)
    offset = bh * DK * DV + rk[:, None] * DV + rv[None, :]
    mask = (rk[:, None] < DK) & (rv[None, :] < DV)
    state = tl.load(H0 + offset, mask, 0).to(tl.float32) * gate
    predicted = tl.sum(k[:, None] * state, axis=0)
    residual = (v - predicted) * beta
    state = state + k[:, None] * residual[None, :]
    output = tl.sum(q[:, None] * state, axis=0) * (DK ** -0.5)
    tl.store(H1 + offset, state, mask)
    tl.store(OUT + bh * DV + rv, output, rv < DV)


def fused_decode(q, k, v, g, beta, initial_state, value_tile=16, num_warps=4):
    if q.device.type != "cuda" or q.shape[1] != 1:
        raise ValueError("fused_decode requires CUDA and T=1")
    if not all(x.is_contiguous() for x in (q, k, v, g, beta, initial_state)):
        raise ValueError("fused_decode requires contiguous inputs")
    batch, _, key_heads, key_dim = q.shape
    heads, value_dim = v.shape[2:]
    if heads % key_heads or k.shape != q.shape:
        raise ValueError("Invalid grouped Q/K heads")
    if initial_state.shape != (batch, heads, key_dim, value_dim):
        raise ValueError("Invalid state shape")
    if g.shape != (batch, 1, heads) or beta.shape != g.shape:
        raise ValueError("Invalid gate shapes")
    if initial_state.dtype != torch.float32:
        raise ValueError("Recurrent state must be FP32")
    output = torch.empty((batch, 1, heads, value_dim), dtype=v.dtype, device=v.device)
    state = torch.empty_like(initial_state)
    _decode[(batch * heads, triton.cdiv(value_dim, value_tile))](
        q, k, v, g, beta, initial_state, output, state,
        key_heads, heads, key_dim, value_dim,
        triton.next_power_of_2(key_dim), value_tile, num_warps=num_warps,
    )
    return output, state
