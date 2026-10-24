import argparse
import subprocess
import sys
from pathlib import Path

from data_utils import DATASET_CHOICES
from models.model_factory import MODEL_CHOICES


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the paper experiment matrix.")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=["DRIVE"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["unet", "attention_unet", "res_attention_unet", "improved_attention_unet"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument(
        "--selection-metric",
        choices=("soft_dice", "hard_dice", "validation_loss"),
        default="soft_dice",
    )
    parser.add_argument("--loss", choices=("bce", "bce_dice"), default="bce")
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--calibrate-threshold", action="store_true")
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--result-root", type=Path, default=PROJECT_ROOT / "result")
    parser.add_argument("--weight-dir", type=Path, default=PROJECT_ROOT / "weight_files")
    parser.add_argument("--execute", action="store_true", help="Run commands; otherwise only print the plan.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_text(command):
    return subprocess.list2cmdline([str(item) for item in command])


def main():
    args = parse_args()
    commands = []
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                result_file = args.result_root / dataset / model / f"seed_{seed}" / "metrics.json"
                best_result_file = args.run_root / dataset / model / f"seed_{seed}" / "best_result.json"
                train = [
                    sys.executable,
                    "-u",
                    str(PROJECT_ROOT / "train.py"),
                    "--dataset", dataset,
                    "--model", model,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--validation-ratio", str(args.validation_ratio),
                    "--workers", str(args.workers),
                    "--patience", str(args.patience),
                    "--min-epochs", str(args.min_epochs),
                    "--selection-metric", args.selection_metric,
                    "--loss", args.loss,
                    "--bce-weight", str(args.bce_weight),
                    "--dice-weight", str(args.dice_weight),
                    "--run-root", str(args.run_root),
                    "--weight-dir", str(args.weight_dir),
                    "--deterministic",
                ]
                if args.overwrite:
                    train.append("--overwrite")
                test = [
                    sys.executable,
                    "-u",
                    str(PROJECT_ROOT / "test.py"),
                    "--dataset", dataset,
                    "--model", model,
                    "--seed", str(seed),
                    "--weight-dir", str(args.weight_dir),
                    "--result-root", str(args.result_root),
                    "--run-root", str(args.run_root),
                    "--workers", str(args.workers),
                    "--save-probabilities",
                ]
                if args.calibrate_threshold:
                    test.extend(
                        [
                            "--calibrate-threshold",
                            "--threshold-min", str(args.threshold_min),
                            "--threshold-max", str(args.threshold_max),
                            "--threshold-step", str(args.threshold_step),
                        ]
                    )
                commands.append((train, test, result_file, best_result_file))

    print(f"Planned runs: {len(commands)} training + {len(commands)} evaluation commands")
    for train, test, result_file, best_result_file in commands:
        status = "[complete]" if result_file.exists() else "[pending]"
        if best_result_file.exists() and not result_file.exists():
            status = "[trained; evaluation pending]"
        print(status)
        print(command_text(train))
        print(command_text(test))
    if not args.execute:
        print("Dry run only. Add --execute to start the experiment matrix.")
        return

    completed = 0
    for index, (train, test, result_file, best_result_file) in enumerate(commands, start=1):
        if result_file.exists():
            completed += 1
            print(f"[{index}/{len(commands)}] Already complete, skipping: {result_file}", flush=True)
            continue
        print(f"[{index}/{len(commands)}] Starting experiment", flush=True)
        if not best_result_file.exists():
            # A directory without best_result.json is an interrupted run. Restart it cleanly.
            run_log = best_result_file.parent / "training_log.csv"
            if run_log.exists() and "--overwrite" not in train:
                train.append("--overwrite")
            subprocess.run(train, cwd=PROJECT_ROOT, check=True)
        else:
            print("Training already complete; running held-out evaluation only.", flush=True)
        subprocess.run(test, cwd=PROJECT_ROOT, check=True)
        completed += 1
        print(f"Progress: {completed}/{len(commands)} experiments complete", flush=True)


if __name__ == "__main__":
    main()
