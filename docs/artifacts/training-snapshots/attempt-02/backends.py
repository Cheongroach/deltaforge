# SPDX-License-Identifier: Apache-2.0
"""Functional GDN backend contract and reversible Transformers 5.2 adapter."""
from collections import OrderedDict
from contextlib import contextmanager
import importlib.metadata
import math

import torch

from .ops import GDN, recurrent


class GDNBackend:
    """Sequence-major, log-decay GDN; state is functional FP32 [B,H,K,V].

    Common preparation is included in each call. Normalization matches the
    Transformers 5.2 fallback, including its input-dtype arithmetic and epsilon.
    No packed sequences, arbitrary beta transforms, or silent fallback.
    """

    def __init__(self, name="deltaforge", chunk_size=32, cache_size=8):
        if name not in ("torch", "deltaforge", "compiled", "fla", "recurrent"):
            raise ValueError(f"Unknown GDN backend: {name}")
        if chunk_size < 1 or cache_size < 1:
            raise ValueError("chunk_size and cache_size must be positive")
        self.name = name
        self.chunk_size = chunk_size
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.calls = self.cache_hits = self.cache_misses = 0
        self.kernel = None
        if name == "fla":
            if importlib.metadata.version("fla-core") != "0.4.1":
                raise RuntimeError("This adapter is verified against fla-core==0.4.1")
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            self.kernel = chunk_gated_delta_rule
        elif name == "torch":
            from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
            self.kernel = torch_chunk_gated_delta_rule

    def __call__(self, q, k, v, g, beta, scale=None, initial_state=None,
                 output_final_state=False, use_qk_l2norm_in_kernel=False,
                 cu_seqlens=None, state_v_first=False, **kwargs):
        if kwargs:
            raise NotImplementedError(f"Unsupported options: {sorted(kwargs)}")
        if cu_seqlens is not None:
            raise NotImplementedError("Packed sequences are not supported by this adapter")
        if q.ndim != 4 or q.shape != k.shape or v.ndim != 4:
            raise ValueError("Q/K [B,T,H_k,K] and V [B,T,H_v,V] are required")
        if (q.shape[1] == 0 or q.shape[2] == 0 or v.shape[2] == 0
                or q.shape[-1] == 0 or v.shape[-1] == 0
                or v.shape[:2] != q.shape[:2] or v.shape[2] % q.shape[2]):
            raise ValueError("Invalid sequence or grouped head dimensions")
        if g.shape != v.shape[:3] or beta.shape != g.shape:
            raise ValueError("g and beta must be [B,T,H_v]")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("Q/K/V must have matching floating dtype")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("Unsupported Q/K/V dtype")
        if any(x.device != q.device for x in (k, v, g, beta)):
            raise ValueError("Inputs must share a device")
        if self.name == "fla" and (q.device.type != "cuda"
                                  or q.dtype not in (torch.float16, torch.bfloat16)):
            raise ValueError("Pinned FLA baseline requires CUDA FP16/BF16")
        scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        if use_qk_l2norm_in_kernel:
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        repeats = v.shape[2] // q.shape[2]
        if repeats != 1:
            q, k = (x.repeat_interleave(repeats, dim=2) for x in (q, k))
        state_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        state_shape = (q.shape[0], v.shape[2], q.shape[-1], v.shape[-1])
        if initial_state is None:
            state = torch.zeros(state_shape, dtype=state_dtype, device=q.device)
        else:
            state = initial_state.transpose(-1, -2) if state_v_first else initial_state
            if state.shape != state_shape or state.device != q.device:
                raise ValueError("Incorrect state shape or device")
            if state.dtype != state_dtype:
                raise ValueError(f"State must have dtype {state_dtype}")
        self.calls += 1
        if self.name == "fla":
            output, final = self.kernel(
                q, k, v, g, beta, scale=scale, initial_state=state,
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=False)
        else:
            # Primitives own the default scale. Adjust only after normalization.
            q = q * (scale * q.shape[-1] ** 0.5)
            if self.name == "torch":
                output, final = self.kernel(
                    q, k, v, g, beta, chunk_size=self.chunk_size,
                    initial_state=state, output_final_state=output_final_state)
            else:
                args = (q, k, v, g, beta, state)
                if self.name == "recurrent":
                    output, final = recurrent(*args)
                else:
                    operation = GDN(self.chunk_size)
                    if self.name == "compiled":
                        key = tuple((tuple(x.shape), tuple(x.stride()), x.dtype,
                                     x.device, x.requires_grad) for x in args)
                        if key in self.cache:
                            self.cache_hits += 1
                            operation = self.cache.pop(key)
                        else:
                            self.cache_misses += 1
                            operation = torch.compile(operation, fullgraph=True, dynamic=False)
                        self.cache[key] = operation
                        if len(self.cache) > self.cache_size:
                            self.cache.popitem(last=False)
                    output, final = operation(*args)
                if not output_final_state:
                    final = None
        if final is not None and state_v_first:
            final = final.transpose(-1, -2)
        return output, final

    def stats(self):
        return {"backend": self.name, "calls": self.calls, "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses, "cache_entries": len(self.cache)}


@contextmanager
def use_backend(model, backend="deltaforge", chunk_size=32):
    """Temporarily replace only per-layer GDN functions; restore on all exits.

    This version adapter intentionally refuses unverified Transformers releases.
    It does not mutate weights, model structure, convolution, norm, or full attention.
    """
    if importlib.metadata.version("transformers") != "5.2.0":
        raise RuntimeError("Qwen3.5 layer adapter requires transformers==5.2.0")
    operation = backend if isinstance(backend, GDNBackend) else GDNBackend(backend, chunk_size)
    layers = [layer for layer in model.modules()
              if type(layer).__name__ == "Qwen3_5GatedDeltaNet"
              and type(layer).__module__.startswith("transformers.models.qwen3_5.")]
    if not layers:
        raise ValueError("No supported Qwen3.5 GDN layers found")
    saved = [(layer, layer.chunk_gated_delta_rule, layer.recurrent_gated_delta_rule)
             for layer in layers]
    try:
        for layer, _, _ in saved:
            layer.chunk_gated_delta_rule = operation
            layer.recurrent_gated_delta_rule = operation
        yield operation
    finally:
        for layer, chunk, token in saved:
            layer.chunk_gated_delta_rule = chunk
            layer.recurrent_gated_delta_rule = token
