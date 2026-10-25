# SPDX-License-Identifier: Apache-2.0
"""Record CPU numerical/framework gates with their exact source revision."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import unittest

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sandbox-exit", action="store_true",
                        help="flush and bypass interpreter teardown in restricted /proc sandboxes")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.time()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
                               for name in ("deltaforge.validation", "deltaforge.test_backends"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    value = {"success": result.wasSuccessful(), "tests": result.testsRun,
             "failures": len(result.failures), "errors": len(result.errors),
             "skipped": len(result.skipped), "elapsed_seconds": time.time() - started,
             "versions": {name: importlib.metadata.version(name)
                          for name in ("torch", "transformers")},
             "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))},
             "scope": "CPU only; excludes FLA, CUDA, compiled backward and pretrained training"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    status = 0 if result.wasSuccessful() else 1
    if args.sandbox_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
