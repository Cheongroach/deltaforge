# SPDX-License-Identifier: Apache-2.0
"""Capture a small, reproducible operator profile, separate from latency benchmarks."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import Counter
from pathlib import Path

import torch

from .ops import GDN, recurrent
from .validation import inputs, verify_pair


def merge_duration(intervals):
    """Union of intervals, rather than summing overlapping GPU streams."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + end - start


def summarize_trace(trace, device):
    events = trace.get("traceEvents", [])
    phases = [
        event for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and event.get("name", "").startswith("deltaforge/")
        and event.get("dur", 0) > 0
    ]
    if not phases:
        raise RuntimeError("Profile contains no DeltaForge phase annotations")
    low = min(event["ts"] for event in phases)
    high = max(event["ts"] + event["dur"] for event in phases)
    timed = [
        event for event in events
        if event.get("ph") == "X" and event.get("dur", 0) > 0
        and low <= event.get("ts", -1) <= high
    ]
    kernels = [event for event in timed if "kernel" in event.get("cat", "").lower()]
    cpu = [event for event in timed if event.get("cat") == "cpu_op"]
    categories = Counter(event.get("cat", "other") for event in timed)
    # Preserve genuine positions/durations. No synthetic GPU rows are added.
    lanes = {}
    timeline = []
    for event in timed:
        category = event.get("cat", "other")
        if category not in ("cpu_op", "user_annotation", "kernel", "cuda_runtime", "cuda_driver", "gpu_memcpy"):
            continue
        lane_key = (category, str(event.get("pid")), str(event.get("tid")))
        if lane_key not in lanes:
            lanes[lane_key] = f"{category} / lane {len(lanes) + 1}"
        timeline.append({
            "name": event["name"], "category": category, "lane": lanes[lane_key],
            "start_us": event["ts"] - low, "duration_us": event["dur"],
        })
    return {
        "device": device, "window_us": high - low,
        "activities": ["CPU", "CUDA"] if device == "cuda" else ["CPU"],
        "captured_gpu_kernels": len(kernels), "cpu_operator_events": len(cpu),
        "gpu_trace_available": bool(kernels),
        "gpu_interval_union_us": merge_duration([
            (event["ts"], event["ts"] + event["dur"]) for event in kernels
        ]),
        "category_counts": dict(categories),
        "phases": [{"name": event["name"], "start_us": event["ts"] - low,
                    "duration_us": event["dur"]} for event in phases],
        "timeline": timeline,
        "limitations": [
            "Profile instrumentation changes timing; use benchmark.py for latency claims.",
            "CPU operator durations nest and must not be summed as elapsed time.",
            "GPU interval union is timeline coverage, not SM occupancy or HBM utilization.",
            "CPU-only captures do not contain GPU kernels.",
        ],
    }


def capture(device, output, iterations=3):
    output.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1 if device == "cpu" else 4)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    dimensions = dict(key_heads=4, value_heads=4, key_dim=32, value_dim=32)
    prefill = inputs(batch=1, length=128, device=device, **dimensions)
    decode = inputs(batch=4, length=1, device=device,
                    key_heads=16, value_heads=16, key_dim=128, value_dim=128,
                    dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    eager = GDN(32)
    functions = [("prefill_eager", lambda: eager(*prefill)),
                 ("decode_torch", lambda: recurrent(*decode))]
    correctness = {"prefill": verify_pair(eager(*prefill), recurrent(*prefill))}
    compiler = None
    if device == "cuda":
        from .compiler import compile_gdn
        from .cuda_decode import fused_decode
        compiled, compiler, _ = compile_gdn(prefill, 32)
        functions.extend([
            ("prefill_compiled", lambda: compiled(*prefill)),
            ("decode_triton", lambda: fused_decode(*decode, value_tile=16, num_warps=4)),
        ])
        correctness["compiled"] = verify_pair(compiled(*prefill), recurrent(*prefill))
        correctness["decode"] = verify_pair(fused_decode(*decode), recurrent(*decode), decode[0].dtype)
    backward_args = tuple(value.detach().requires_grad_() for value in prefill)

    def backward():
        for value in backward_args:
            value.grad = None
        out, state = eager(*backward_args)
        (out.square().mean() + state.square().mean()).backward()

    # All compilation and warmup happen outside the measured profile.
    for _, function in functions:
        for _ in range(4):
            function()
    backward()
    sync()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True, with_stack=False,
    ) as profiler:
        for iteration in range(iterations):
            for label, function in functions:
                with torch.profiler.record_function(f"deltaforge/{label}/{iteration}"):
                    function()
                    sync()
            with torch.profiler.record_function(f"deltaforge/pytorch_forward_backward/{iteration}"):
                backward()
                sync()
    trace_path = output / "trace.json"
    profiler.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    summary = summarize_trace(trace, device)
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).parent.glob("*.py")
    }
    versions = {}
    for package in ("torch", "triton", "torch-npu", "torchair"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    summary.update({
        "capture_date": time.strftime("%Y-%m-%d"),
        "fixture": "seeded_random_operator_inputs", "seed": 20261025,
        "end_to_end_model_training": False, "iterations": iterations,
        "prefill_shape": {"B": 1, "T": 128, "H": 4, "K": 32, "V": 32, "chunk": 32},
        "decode_shape": {"B": 4, "T": 1, "H": 16, "K": 128, "V": 128},
        "synchronize_inside_phase": device == "cuda",
        "versions": versions, "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "source_sha256": source_hashes, "correctness": correctness, "compiler": compiler,
        "operator_table": [
            {"name": entry.key, "calls": entry.count,
             "self_cpu_us": entry.self_cpu_time_total,
             "total_cpu_us": entry.cpu_time_total,
             "self_device_us": getattr(entry, "self_device_time_total", 0),
             "self_cpu_memory_bytes": entry.self_cpu_memory_usage}
            for entry in profiler.key_averages()
        ],
    })
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    # gzip is a deliverable format; absolute host paths/IDs are removed on public export.
    with gzip.open(output / "trace.json.gz", "wb") as compressed:
        compressed.write(trace_path.read_bytes())
    print("PROFILE_CAPTURE " + json.dumps({
        "device": device, "gpu_kernels": len([
            event for event in summary["timeline"] if event["category"] == "kernel"
        ]), "cpu_events": summary["cpu_operator_events"],
        "trace_bytes": trace_path.stat().st_size, "summary": str(output / "summary.json"),
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 20:
        parser.error("iterations must be between 1 and 20")
    capture(args.device, args.output, args.iterations)
