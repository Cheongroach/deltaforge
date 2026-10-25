# SPDX-License-Identifier: Apache-2.0
"""Real-device measurements; seeded random tensors are operator fixtures, not model data."""
import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
import traceback
import unittest
from pathlib import Path

import torch

from .compiler import compile_gdn
from .ops import GDN, recurrent
from .validation import NumericalTests, device_validation, inputs, verify_pair


def emit(row, root=None):
    print("DELTAFORGE " + json.dumps(row, ensure_ascii=False), flush=True)
    if root is not None:
        with (root / "events.jsonl").open("a") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def latency(function, args, device, repeats=7, iterations=5):
    api = getattr(torch, device)
    for _ in range(4):
        function(*args)
    api.synchronize()
    wall, event = [], []
    for _ in range(repeats):
        start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
        api.synchronize()
        clock = time.perf_counter()
        start.record()
        for _ in range(iterations):
            function(*args)
        end.record()
        api.synchronize()
        wall.append((time.perf_counter() - clock) * 1000 / iterations)
        event.append(start.elapsed_time(end) / iterations)
    return {
        "wall_ms_median": statistics.median(wall),
        "wall_ms_min": min(wall), "wall_ms_max": max(wall),
        "device_ms_median": statistics.median(event),
        "wall_ms_samples": wall, "device_ms_samples": event,
        "warmup": 4, "repeats": repeats, "iterations": iterations,
    }


def record(function, args, expected, label, shape, device, root, **extra):
    checked = verify_pair(function(*args), expected, args[0].dtype)
    row = {"event": "measurement", "implementation": label, "shape": shape,
           "device": device, "correctness": checked, **extra}
    row.update(latency(function, args, device))
    emit(row, root)
    return row


def failure(label, error, root, **extra):
    emit({"event": "unsupported_or_failed", "implementation": label,
          "error_type": type(error).__name__, "error": str(error)[:1800], **extra}, root)
    with (root / "failures.txt").open("a") as out:
        out.write(label + "\n" + traceback.format_exc() + "\n")


def cuda_experiments(root):
    from .cuda_decode import fused_decode

    for batch in (1, 4, 16):
        args = inputs(batch=batch, length=1, key_heads=16, value_heads=16,
                      key_dim=128, value_dim=128, device="cuda", dtype=torch.bfloat16)
        expected = recurrent(*(x.cpu() for x in args))
        expected = tuple(x.to("cuda") for x in expected)
        shape = {"B": batch, "T": 1, "Hk": 16, "Hv": 16, "K": 128, "V": 128, "dtype": "bfloat16"}
        record(recurrent, args, expected, "torch_token_recurrence", shape, "cuda", root)
        candidates = []
        for tile in (8, 16, 32, 64):
            for warps in (4, 8):
                def implementation(*xs, tile=tile, warps=warps):
                    return fused_decode(*xs, value_tile=tile, num_warps=warps)
                try:
                    row = record(implementation, args, expected, "triton_fused_decode", shape,
                                 "cuda", root, value_tile=tile, num_warps=warps)
                    candidates.append(row)
                except (RuntimeError, AssertionError, ValueError) as error:
                    failure("triton_fused_decode", error, root, value_tile=tile, num_warps=warps)
        if candidates:
            best = min(candidates, key=lambda row: row["wall_ms_median"])
            emit({"event": "tuning_winner", "shape": shape, "device": "cuda",
                  "value_tile": best["value_tile"], "num_warps": best["num_warps"],
                  "wall_ms_median": best["wall_ms_median"]}, root)
    # Non-power-of-two dimensions and grouped heads exercise masks and head mapping.
    args = inputs(batch=2, length=1, key_heads=2, value_heads=4,
                  key_dim=37, value_dim=53, device="cuda", dtype=torch.bfloat16)
    verify_pair(fused_decode(*args), recurrent(*args), torch.bfloat16)
    # Repeated functional cache handoff, with no cumulative state truncation.
    sequence = inputs(batch=1, length=128, key_heads=4, value_heads=4,
                      key_dim=64, value_dim=64, device="cuda", dtype=torch.bfloat16)
    state, outputs = sequence[5], []
    for index in range(128):
        tokens = tuple(x[:, index:index + 1].contiguous() for x in sequence[:5])
        output, state = fused_decode(*tokens, state)
        outputs.append(output)
    check = verify_pair((torch.cat(outputs, dim=1), state), recurrent(*sequence), torch.bfloat16)
    emit({"event": "decode_handoff_validation", "steps": 128, "odd_dims_verified": True,
          "correctness": check}, root)


def prefill_experiments(device, root):
    # Exact Qwen3.5-2B GDN dimensions, plus small-head settings relevant under TP.
    shapes = [(1, 128, 16, 128), (1, 512, 16, 128), (1, 1024, 4, 128)]
    for batch, length, heads, dimension in shapes:
        args = inputs(batch=batch, length=length, key_heads=heads, value_heads=heads,
                      key_dim=dimension, value_dim=dimension, device=device)
        expected = recurrent(*(x.cpu() for x in args))
        expected = tuple(x.to(device) for x in expected)
        shape = {"B": batch, "T": length, "Hk": heads, "Hv": heads,
                 "K": dimension, "V": dimension, "dtype": "float32"}
        candidates = []
        for chunk in (16, 32, 64):
            function = GDN(chunk)
            try:
                row = record(function, args, expected, "portable_chunk_eager", shape,
                             device, root, chunk_size=chunk)
                candidates.append(row)
            except (RuntimeError, AssertionError, ValueError) as error:
                failure("portable_chunk_eager", error, root, shape=shape, chunk_size=chunk)
        if not candidates:
            continue
        best = min(candidates, key=lambda row: row["wall_ms_median"])
        chunk = best["chunk_size"]
        emit({"event": "tuning_winner", "device": device, "shape": shape,
              "chunk_size": chunk, "wall_ms_median": best["wall_ms_median"]}, root)
        try:
            record(GDN(chunk, semantic_solve=True), args, expected, "native_triangular_solve",
                   shape, device, root, chunk_size=chunk)
        except (RuntimeError, AssertionError, ValueError, NotImplementedError) as error:
            failure("native_triangular_solve", error, root, shape=shape, chunk_size=chunk)
        try:
            function, info, graph_code = compile_gdn(args, chunk)
            (root / f"lowered-{length}-{heads}.py").write_text(graph_code)
            row = record(function, args, expected, "fx_lowered_compiled", shape, device, root,
                         chunk_size=chunk, compiler=info)
            # Retest every compiled result; no automatic fallback counted as compilation.
            emit({"event": "compiler_speedup", "device": device, "shape": shape,
                  "baseline": "portable_chunk_eager_same_chunk",
                  "wall_speedup": best["wall_ms_median"] / row["wall_ms_median"]}, root)
        except (RuntimeError, AssertionError, ValueError, ImportError, NotImplementedError) as error:
            failure("fx_lowered_compiled", error, root, shape=shape, chunk_size=chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "npu"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, device = args.output, args.device
    root.mkdir(parents=True, exist_ok=True)
    if device == "npu":
        import torch_npu  # noqa: F401
    getattr(torch, device).set_device(0)
    torch.set_num_threads(4)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    emit({"event": "benchmark_start", "device": device, "torch": torch.__version__,
          "device_name": getattr(torch, device).get_device_name(0),
          "source_sha256": hashes, "workload": "seeded_random_operator_fixtures",
          "end_to_end_model_benchmark": False, "state_dtype": "float32"}, root)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(NumericalTests))
    if not result.wasSuccessful():
        raise RuntimeError("CPU regression gate failed inside device container")
    emit({"event": "cpu_regression", "tests": result.testsRun, "passed": True}, root)
    for dtype in (torch.float32, torch.bfloat16):
        checked = device_validation(device, dtype)
        emit({"event": "device_correctness", "device": device, "dtype": str(dtype),
              "cases": checked, "passed": True}, root)
    if device == "cuda":
        cuda_experiments(root)
    prefill_experiments(device, root)
    emit({"event": "benchmark_complete", "device": device, "terminal": True}, root)


if __name__ == "__main__":
    main()
