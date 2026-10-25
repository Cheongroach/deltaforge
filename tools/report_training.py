# SPDX-License-Identifier: Apache-2.0
"""Recompute a compact summary and check the terminal training evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text())


def summarize(attempt, lock):
    status = load(attempt / "run-status.json")
    if status["status"] != "done" or status["pod_exit_statuses"] != [0]:
        raise ValueError("Training run is not terminal with exit 0")
    evidence = attempt / "evidence"
    cpu = load(evidence / "cpu-gates.json")
    if not cpu["success"] or cpu["tests"] < 14 or cpu["skipped"]:
        raise ValueError("CPU gate evidence failed")
    gates = load(evidence / "model-gate/model-gates.json")
    accepted = {row["backend"] for row in gates if row["gate_passed"]}
    if not {"torch", "deltaforge", "compiled"}.issubset(accepted):
        raise ValueError("Required model gates failed")
    protocol = lock["protocol"]
    for row in gates:
        if row["backend"] not in accepted or row["backend"] == "torch":
            continue
        checks = row["checks"]
        if (not checks["gradients_complete_and_finite"] or not checks["logits"]["finite"]
                or checks["logits"]["relative_l2"] > protocol["model_logits_relative_l2_max"]
                or checks["parameter_gradient_relative_l2"] > protocol["gradient_relative_l2_max"]
                or checks["loss_abs_difference"] > protocol["model_loss_abs_max"]):
            raise ValueError("Recorded gate disagrees with the pinned thresholds")
    training = []
    hashes = {}
    validation = {}
    initialization = set()
    for backend in sorted(accepted):
        hashes[backend] = set()
        for repetition in range(protocol["independent_process_repetitions"]):
            row = load(evidence / f"train-{backend}-{repetition}"
                       / f"training-{backend}-{repetition}.json")
            initialization.add(row["initial_adapter_sha256"])
            if (len(row["steps"]) != protocol["train_steps"]
                    or row["initial_adapter_sha256"] == row["final_adapter_sha256"]
                    or row["data_sha256"] != lock["data_sha256"]):
                raise ValueError("Training budget, parameter update or data hash failed")
            hashes[backend].add(row["final_adapter_sha256"])
            validation[backend, repetition] = row["validation_loss_after"]
            steps = [s for s in row["steps"] if not s["warmup"]]
            times = [s["seconds"] for s in steps]
            training.append({"backend": backend, "repetition": repetition,
                "measured_steps": len(steps),
                "effective_tokens": sum(s["effective_tokens"] for s in steps),
                "step_median_ms": statistics.median(times) * 1000,
                "step_min_ms": min(times) * 1000, "step_max_ms": max(times) * 1000,
                "effective_tokens_per_second": sum(s["effective_tokens"] for s in steps) / sum(times),
                "peak_allocated_gib": max(s["peak_allocated_bytes"] for s in steps) / 2**30,
                "validation_loss_before": row["validation_loss_before"],
                "validation_loss_after": row["validation_loss_after"],
                "final_adapter_sha256": row["final_adapter_sha256"]})
    if len(initialization) != 1 or any(len(values) != 1 for values in hashes.values()):
        raise ValueError("Initialization or within-attempt repeatability differs")
    for (backend, repetition), value in validation.items():
        if abs(value - validation["torch", repetition]) > protocol["validation_loss_backend_abs_max"]:
            raise ValueError("Final validation loss comparison failed")
    kernels = []
    for path in sorted(evidence.glob("kernel-*/*.json")):
        for row in load(path):
            samples = row["samples"]
            kernels.append({"repetition": int(path.parent.name.split("-")[-1]),
                "backend": row["backend"], "length": row["length"],
                "gate_passed": row["gate_passed"],
                "forward_median_ms": statistics.median(s["forward_ms"] for s in samples) if samples else None,
                "backward_median_ms": statistics.median(s["backward_ms"] for s in samples) if samples else None,
                "first_forward_backward_ms": row.get("first_forward_backward_ms"),
                "saved_tensor_bytes": row.get("saved_tensor_bytes", [])})
    return {"status": status, "reference": "torch adapter, not native default",
            "cpu_tests": cpu["tests"], "accepted": sorted(accepted),
            "rejected": [r["backend"] for r in gates if not r["gate_passed"]],
            "model_gates": [{"backend": r["backend"], "gate_passed": r["gate_passed"],
                             "checks": {k: v for k, v in r.get("checks", {}).items()
                                        if k != "largest_gradient_error_contributors"}} for r in gates],
            "training": training, "kernels": kernels,
            "scope": "Single H20, BF16 Qwen3.5-2B LoRA rank8, fixed 256-token bucket; two processes sharing disk caches"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, default=Path("training-results/attempt-03"))
    parser.add_argument("--lock", type=Path, default=Path("training-assets/lock.json"))
    parser.add_argument("--output", type=Path, default=Path("training-results/summary.json"))
    args = parser.parse_args()
    value = summarize(args.attempt, load(args.lock))
    value["evidence_sha256"] = {
        str(p.relative_to(args.attempt)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.attempt.rglob("*.json"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"verified": True, "cpu_tests": value["cpu_tests"],
                      "training_arms": len(value["training"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
