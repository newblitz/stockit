"""Train ten independent seeds and aggregate their held-out test metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.evaluation import aggregate_runs, paired_t_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible multi-seed paper-model experiments.")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/proposed"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--compare-summary", type=Path, help="JSON with per_seed results from a baseline/ablation.")
    args, train_args = parser.parse_known_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict] = []
    for seed in args.seeds:
        directory = args.output_root / f"seed_{seed}"
        subprocess.run([sys.executable, "train.py", "--seed", str(seed), "--checkpoint-dir", str(directory), *train_args], check=True)
        per_seed.append(json.loads((directory / "metrics.json").read_text()))
    summary: dict = {"seeds": args.seeds, "per_seed": per_seed,
                      "aggregate": aggregate_runs(per_seed, ("test_accuracy", "test_mcc", "test_loss"))}
    if args.compare_summary:
        comparison = json.loads(args.compare_summary.read_text())["per_seed"]
        for metric in ("test_accuracy", "test_mcc"):
            summary[f"{metric}_paired_t_test"] = paired_t_test(
                [run[metric] for run in per_seed], [run[metric] for run in comparison]
            )
    output = args.output_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
