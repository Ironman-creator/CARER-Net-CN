"""Reject stale artifacts before the PowerShell runner skips a completed job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--min-epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    return parser.parse_args()


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def split_sets(payload):
    return set(payload["train"]), set(payload["validation"])


def main():
    args = parse_args()
    config = read_json(args.run_dir / "config.json")
    result = read_json(args.run_dir / "best_result.json")
    observed_split = read_json(args.run_dir / "split.json")
    expected_split = read_json(args.expected_split)
    expected = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "image_size": 512,
        "input_mode": "grayscale",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 0.0001,
        "loss": "bce_dice",
        "bce_weight": 1.0,
        "dice_weight": 1.0,
        "selection_metric": "soft_dice",
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "deterministic": True,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if split_sets(observed_split) != split_sets(expected_split):
        mismatches["split"] = "run split differs from frozen reference split"
    if not args.checkpoint.is_file():
        mismatches["checkpoint"] = f"missing {args.checkpoint}"
    if result.get("best_epoch") is None:
        mismatches["best_result"] = "missing best_epoch"
    if mismatches:
        raise ValueError(
            "Completed artifact is incompatible with the frozen stage-2 protocol; "
            "inspect it or rerun explicitly with -Overwrite. "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    print(f"Compatible completed run: {args.dataset}/{args.model}/seed_{args.seed}")


if __name__ == "__main__":
    main()
