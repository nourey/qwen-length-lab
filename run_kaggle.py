#!/usr/bin/env python3
"""Packaging entry point: unchanged 4-prompt smoke gate, then 48-prompt pilot."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_pipeline(output_root):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_dir = output_root / "v1-smoke-qwen3-8b"
    pilot_dir = output_root / "v1-pilot-qwen3-8b"

    def execute(*args):
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)

    def experiment(*args):
        execute(ROOT / "experiment.py", *args)

    common = ["--model-size", "8b", "--device", "0", "--cap", "1536", "--seed", "20260916"]
    execute("-m", "unittest", "discover", "-s", "tests", "-v")
    experiment("preflight", "--model-size", "8b", "--device", "0")
    experiment("run", "--mode", "smoke", "--out", smoke_dir, *common)
    counts = json.loads((smoke_dir / "metrics.json").read_text())["counts"]
    if counts["planned"] != 4 or counts["generation_successes"] != 4 or counts["prediction_failures"] != 0:
        raise RuntimeError(f"Smoke gate failed. Inspect {smoke_dir}; pilot was not started.")
    revision = json.loads((smoke_dir / "manifest.json").read_text())["resolved_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Smoke manifest must contain an immutable model commit SHA")
    experiment("run", "--mode", "pilot", "--out", pilot_dir, "--revision", revision, *common)
    execute(ROOT / "plot_results.py", pilot_dir)
    print(f"Pilot CSV/JSONL/metrics: {pilot_dir}", flush=True)
    return pilot_dir


def main():
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Use this entry point on Kaggle; experiment.py remains usable on compatible CUDA hosts.")
    os.environ.setdefault("HF_HOME", "/kaggle/temp/qwen-length-model-cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    run_pipeline(os.environ.get("QWEN_LAB_RUN_ROOT", "/kaggle/working/qwen-length-runs"))


if __name__ == "__main__":
    main()
