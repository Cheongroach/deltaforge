# SPDX-License-Identifier: Apache-2.0
"""Exact Gated DeltaNet primitives. State layout is [B, H_v, K, V]."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def inverse_unit_lower(lower: torch.Tensor) -> torch.Tensor:
    """Finite doubling; exact in real arithmetic because strictly lower L**C=0."""
    size = lower.shape[-1]
    eye = torch.eye(size, device=lower.device, dtype=lower.dtype)
    power = -lower
    inverse = eye + power
    for _ in range(1, math.ceil(math.log2(size))):
        power = power @ power
        inverse = inverse + power @ inverse
    return inverse if size > 1 else eye.expand_as(lower)


@torch.library.custom_op("deltaforge::unit_lower_solve", mutates_args=())
def unit_lower_solve(lower: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Semantic IR op: lower is STRICTLY lower, diagonal is supplied implicitly."""
    eye = torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype)
    return torch.linalg.solve_triangular(eye + lower, rhs, upper=False, unitriangular=True)


@unit_lower_solve.register_fake
def _fake_solve(lower, rhs):
    return torch.empty_like(rhs)


def _prepare(q, k, v, g, beta, lengths=None):
    # Transpose into head-major layout. Expand GQA heads in the same order as HF.
    repeats = v.shape[2] // k.shape[2]
    q = q.repeat_interleave(repeats, dim=2).transpose(1, 2).float()
    k = k.repeat_interleave(repeats, dim=2).transpose(1, 2).float()
    v = v.transpose(1, 2).float()
    g, beta = g.transpose(1, 2).float(), beta.transpose(1, 2).float()
    if lengths is not None:
        valid = torch.arange(q.shape[2], device=q.device)[None, :] < lengths[:, None]
        valid = valid[:, None, :]
        q, k, v = (x * valid[..., None] for x in (q, k, v))
        g, beta = g * valid, beta * valid  # Padding is an identity state transition.
    return q, k, v, g, beta


def recurrent(q, k, v, g, beta, initial_state, lengths=None):
    """Independent token recurrence, used as the FP32 correctness oracle."""
    dtype = v.dtype
    q, k, v, g, beta = _prepare(q, k, v, g, beta, lengths)
    state = initial_state.float()
    output = []
    scale = q.shape[-1] ** -0.5
    for token in range(q.shape[2]):
        state = state * g[:, :, token].exp()[..., None, None]
        kt, vt = k[:, :, token], v[:, :, token]
        predicted = (kt[..., None] * state).sum(-2)
        residual = (vt - predicted) * beta[:, :, token, None]
        state = state + kt[..., None] * residual[..., None, :]
        output.append(((q[:, :, token, :, None] * state).sum(-2)) * scale)
    return torch.stack(output, dim=2).transpose(1, 2).to(dtype), state


class GDN(nn.Module):
    """Chunked GDN with a choice of semantic solve or portable matmul lowering.

    Inputs: normalized Q/K [B,T,H_k,K], V [B,T,H_v,V], g/beta [B,T,H_v],
    initial FP32 state [B,H_v,K,V]. g is non-positive LOG decay.
    The caller owns normalization, gate construction, and request-to-state mapping.
    """

    def __init__(self, chunk_size=32, semantic_solve=False):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.semantic_solve = semantic_solve

    def forward(self, q, k, v, g, beta, initial_state, lengths=None):
        dtype, original_length = v.dtype, q.shape[1]
        q, k, v, g, beta = _prepare(q, k, v, g, beta, lengths)
        size = self.chunk_size
        padding = (-original_length) % size
        q, k, v = (F.pad(x, (0, 0, 0, padding)) for x in (q, k, v))
        g, beta = F.pad(g, (0, padding)), F.pad(beta, (0, padding))
        batch, heads, length, key_dim = k.shape
        chunks = length // size
        q = q.reshape(batch, heads, chunks, size, key_dim) * (key_dim ** -0.5)
        k = k.reshape(batch, heads, chunks, size, key_dim)
        v = v.reshape(batch, heads, chunks, size, v.shape[-1])
        g, beta = g.reshape(batch, heads, chunks, size), beta.reshape(batch, heads, chunks, size)
        prefix = g.cumsum(-1)
        # Never divide exp(prefix): prefix can underflow at long context.
        decay = (prefix[..., :, None] - prefix[..., None, :]).clamp(max=0).exp()
        causal = torch.ones(size, size, device=q.device, dtype=torch.bool).tril()
        decay = decay * causal
        lower = (beta[..., :, None] * (k @ k.transpose(-1, -2)) * decay).tril(-1)
        rhs = torch.cat((beta[..., None] * v, (beta * prefix.exp())[..., None] * k), dim=-1)
        factors = (
            unit_lower_solve(lower, rhs)
            if self.semantic_solve
            else inverse_unit_lower(lower) @ rhs
        )
        u, w = factors[..., :v.shape[-1]], factors[..., v.shape[-1]:]
        attention = (q @ k.transpose(-1, -2)) * decay
        k_tail = k * (prefix[..., -1, None] - prefix).exp()[..., None]
        state = initial_state.float()
        outputs = []
        for chunk in range(chunks):
            residual = u[:, :, chunk] - w[:, :, chunk] @ state
            output = (
                prefix[:, :, chunk].exp()[..., None] * (q[:, :, chunk] @ state)
                + attention[:, :, chunk] @ residual
            )
            state = (
                prefix[:, :, chunk, -1].exp()[..., None, None] * state
                + k_tail[:, :, chunk].transpose(-1, -2) @ residual
            )
            outputs.append(output)
        output = torch.cat(outputs, dim=2)[:, :, :original_length]
        return output.transpose(1, 2).to(dtype), state


def validate_inputs(q, k, v, g, beta, state, lengths=None):
    if q.ndim != 4 or q.shape != k.shape or v.ndim != 4:
        raise ValueError("Q/K must match [B,T,H_k,K]; V must be [B,T,H_v,V]")
    if v.shape[:2] != q.shape[:2] or q.shape[1] == 0 or v.shape[2] % q.shape[2]:
        raise ValueError("Invalid batch/sequence or grouped head counts")
    if g.shape != v.shape[:3] or beta.shape != g.shape:
        raise ValueError("g/beta must be [B,T,H_v]")
    if state.shape != (q.shape[0], v.shape[2], q.shape[-1], v.shape[-1]):
        raise ValueError("Incorrect initial state shape")
    tensors = (q, k, v, g, beta, state)
    if any(x.device != q.device for x in tensors):
        raise ValueError("All inputs must be on the same device")
    if not all(torch.isfinite(x).all().item() for x in tensors):
        raise ValueError("Inputs must be finite")
    if g.max().item() > 0 or beta.min().item() < 0 or beta.max().item() > 1:
        raise ValueError("Contract: g <= 0 and 0 <= beta <= 1")
    if lengths is not None:
        if lengths.shape != (q.shape[0],) or lengths.device != q.device:
            raise ValueError("lengths must be [B] on the input device")
        if lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("lengths must be integer")
        if lengths.min().item() < 0 or lengths.max().item() > q.shape[1]:
            raise ValueError("lengths out of range")
