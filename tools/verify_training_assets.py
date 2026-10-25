# SPDX-License-Identifier: Apache-2.0
"""Verify the pinned public model and dataset before running the training probe."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("training-assets"))
    args = parser.parse_args()
    lock = json.loads((args.assets / "lock.json").read_text())
    if sha256(args.assets / "data.json") != lock["data_sha256"]:
        raise ValueError("Dataset SHA256 differs from the training lock")
    checked = []
    for name, record in lock["model"]["files"].items():
        path = args.model / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Model file differs from the training lock: {name}")
        checked.append(name)
    print(json.dumps({"verified": True, "model_files": len(checked),
                      "data_sha256": lock["data_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
