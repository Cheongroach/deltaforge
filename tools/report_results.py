# SPDX-License-Identifier: Apache-2.0
"""Regenerate measured summaries from archived events, without platform access."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / "artifacts" / "runs"
    all_rows, measurements, summaries = [], [], []
    for path in sorted(raw.glob("*.events.jsonl")):
        run = path.name.removesuffix(".events.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        all_rows.extend({"run": run, **row} for row in rows)
        for row in rows:
            if row["event"] != "measurement":
                continue
            shape = row["shape"]
            flat = {"run": run, "device": row["device"], "implementation": row["implementation"],
                    **shape, "chunk_size": row.get("chunk_size"), "value_tile": row.get("value_tile"),
                    "num_warps": row.get("num_warps"), "wall_ms_median": row["wall_ms_median"],
                    "device_ms_median": row["device_ms_median"],
                    "wall_ms_min": row["wall_ms_min"], "wall_ms_max": row["wall_ms_max"],
                    "output_max_abs": row["correctness"]["output"]["max_abs"],
                    "state_max_abs": row["correctness"]["state"]["max_abs"],
                    "compile_seconds": row.get("compiler", {}).get("compile_and_first_execution_seconds"),
                    "export_seconds": row.get("compiler", {}).get("export_and_lower_seconds")}
            measurements.append(flat)
        if not run.endswith("constants"):
            continue
        for row in rows:
            if row["event"] == "measurement" and row["implementation"] == "fx_lowered_compiled":
                base = next(x for x in rows if x["event"] == "measurement"
                            and x["implementation"] == "portable_chunk_eager"
                            and x["shape"] == row["shape"] and x["chunk_size"] == row["chunk_size"])
                delta_ms = base["wall_ms_median"] - row["wall_ms_median"]
                info = row["compiler"]
                total = info["export_and_lower_seconds"] + info["compile_and_first_execution_seconds"]
                summaries.append({"run": run, "kind": "prefill", "shape": row["shape"],
                    "baseline": "portable_chunk_eager_same_chunk", "chunk_size": row["chunk_size"],
                    "baseline_ms": base["wall_ms_median"], "optimized_ms": row["wall_ms_median"],
                    "speedup": base["wall_ms_median"] / row["wall_ms_median"],
                    "compile_seconds": total,
                    "break_even_calls": math.ceil(total * 1000 / delta_ms) if delta_ms > 0 else None,
                    "compiler": info, "correctness": row["correctness"]})
            if row["event"] == "tuning_winner" and row["shape"]["T"] == 1:
                base = next(x for x in rows if x["event"] == "measurement"
                            and x["implementation"] == "torch_token_recurrence" and x["shape"] == row["shape"])
                best = next(x for x in rows if x["event"] == "measurement"
                            and x["implementation"] == "triton_fused_decode" and x["shape"] == row["shape"]
                            and x["value_tile"] == row["value_tile"] and x["num_warps"] == row["num_warps"])
                summaries.append({"run": run, "kind": "decode", "shape": row["shape"],
                    "baseline": "own_torch_token_recurrence", "value_tile": row["value_tile"],
                    "num_warps": row["num_warps"], "baseline_ms": base["wall_ms_median"],
                    "optimized_ms": best["wall_ms_median"],
                    "speedup": base["wall_ms_median"] / best["wall_ms_median"],
                    "correctness": best["correctness"]})
    artifact = ROOT / "artifacts"
    (artifact / "measurements.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n")
    for row in summaries:
        row["recommendation"] = "compiled_candidate" if row["kind"] == "prefill" and row["speedup"] > 1 else (
            "eager" if row["kind"] == "prefill" else "triton_candidate")
    (artifact / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    plan = {"offline_recommendation_only": True, "independent_confirmation_pending": True,
            "rule": "Reject compiled candidates whose measured latency exceeds same-chunk eager",
            "entries": [{key: row[key] for key in ("run", "kind", "shape", "speedup", "recommendation")}
                        for row in summaries]}
    (artifact / "dispatch-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    if measurements:
        with (artifact / "measurements.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(measurements[0]))
            writer.writeheader(); writer.writerows(measurements)
    print("Archived events", len(all_rows), "measurements", len(measurements), "final comparisons", len(summaries))
    for row in summaries:
        print(row["run"], row["kind"], row["shape"], round(row["speedup"], 3))


if __name__ == "__main__":
    main()
