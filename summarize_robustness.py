from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_robustness import CORRUPTIONS


MODELS = ("gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet")
METRICS = ("dice", "sensitivity", "specificity", "iou", "auc_roc", "auc_pr")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(delta, rng, count=10000):
    indices = rng.integers(0, len(delta), size=(count, len(delta)))
    values = delta[indices].mean(axis=1)
    return np.quantile(values, [0.025, 0.975])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260927)
    values = {}
    summary_rows = []
    reference_samples = None
    for corruption in CORRUPTIONS:
        for model in MODELS:
            path = args.result_root / corruption / model / "seed_42" / "per_image_metrics.csv"
            rows = read_csv(path)
            samples = [row["sample"].lower() for row in rows]
            if len(samples) != len(set(samples)):
                raise ValueError(f"Duplicate samples: {path}")
            if reference_samples is None:
                reference_samples = samples
            if samples != reference_samples:
                raise ValueError(f"Sample order mismatch: {path}")
            for metric in METRICS:
                array = np.asarray([float(row[metric]) for row in rows])
                values[(corruption, model, metric)] = array
                summary_rows.append(
                    {
                        "corruption": corruption,
                        "model": model,
                        "metric": metric,
                        "n_images": len(array),
                        "mean": float(array.mean()),
                        "std": float(array.std(ddof=1)),
                    }
                )
    comparison_rows = []
    for corruption in CORRUPTIONS:
        for metric in METRICS:
            for candidate, baseline, label in (
                ("carer_unet", "gray_rgb_control_30_unet", "CARER-control"),
                ("carer_unet", "color_detail_only_unet", "CARER-color"),
            ):
                delta = values[(corruption, candidate, metric)] - values[(corruption, baseline, metric)]
                low, high = bootstrap(delta, rng)
                comparison_rows.append(
                    {
                        "corruption": corruption,
                        "comparison": label,
                        "metric": metric,
                        "n_images": len(delta),
                        "mean_delta": float(delta.mean()),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "positive_fraction": float(np.mean(delta > 0)),
                    }
                )
            for model in MODELS:
                degradation = values[(corruption, model, metric)] - values[("clean", model, metric)]
                low, high = bootstrap(degradation, rng)
                comparison_rows.append(
                    {
                        "corruption": corruption,
                        "comparison": f"{model}:corrupted-clean",
                        "metric": metric,
                        "n_images": len(degradation),
                        "mean_delta": float(degradation.mean()),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "positive_fraction": float(np.mean(degradation > 0)),
                    }
                )
    write_csv(args.output_dir / "FIVES_robustness_summary.csv", summary_rows)
    write_csv(args.output_dir / "FIVES_robustness_pairwise.csv", comparison_rows)
    report = {
        "scope": "exploratory deterministic robustness on the frozen seed-42 checkpoint",
        "models": list(MODELS),
        "corruptions": list(CORRUPTIONS),
        "threshold": 0.5,
        "fov": "derived from clean image and held fixed across corruptions",
        "statistical_unit": "paired FIVES official-test image",
    }
    (args.output_dir / "FIVES_robustness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Robustness summary written to {args.output_dir}")


if __name__ == "__main__":
    main()
