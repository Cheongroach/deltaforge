# SPDX-License-Identifier: Apache-2.0
"""Refresh the offline Lab from actual JSON; never generate benchmark numbers."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, default=ROOT / "artifacts/measurements.json")
    parser.add_argument("--comparisons", type=Path, default=ROOT / "artifacts/summary.json")
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "profiling-lab/assets/data.js")
    args = parser.parse_args()
    profiles = {}
    for device in ("cpu", "cuda"):
        path = getattr(args, device)
        if path is None:
            path = ROOT / f"profiling-lab/assets/{device}-summary.json"
        if path.exists():
            profile = json.loads(path.read_text())
            if not profile.get("timeline") or not profile.get("phases"):
                raise ValueError(f"{device} profile has no timeline/phase data")
            profiles[device] = profile
    if not profiles:
        raise ValueError("Provide at least one genuine CPU/CUDA profile JSON")
    data = {
        "measurements": json.loads(args.measurements.read_text()),
        "summary": json.loads(args.comparisons.read_text()),
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("window.DELTAFORGE_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n")
    print("Updated", args.output, "profiles", sorted(profiles))


if __name__ == "__main__":
    main()
