"""Calibration and image-level failure-detection analysis from saved probabilities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata, spearmanr

from data_utils import discover_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resize_mask(path, size):
    return np.asarray(
        Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST),
        dtype=np.uint8,
    ) > 127


def resize_fov(path, size):
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return gray > (5.0 / 255.0)


def load_probability(path, size):
    image = Image.open(path)
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.integer):
        observed_maximum = int(array.max(initial=0))
        if observed_maximum <= 1:
            maximum = 1
        elif observed_maximum <= 255:
            maximum = 255
        elif observed_maximum <= 65535:
            maximum = 65535
        else:
            maximum = np.iinfo(array.dtype).max
        return array.astype(np.float64) / float(maximum)
    return np.asarray(array, dtype=np.float64)


def binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def main():
    args = parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be >=2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = []
    bin_rows = []
    per_image_rows = []

    for dataset in args.datasets:
        pairs = discover_pairs(args.data_root, dataset, "test")
        pair_index = {key.lower(): (image, mask) for image, mask, key in pairs}
        for model in args.models:
            for seed in args.seeds:
                directory = args.result_root / dataset / model / f"seed_{seed}"
                metrics_rows = read_csv(directory / "per_image_metrics.csv")
                metric_index = {row["sample"].lower(): float(row["dice"]) for row in metrics_rows}
                if set(metric_index) != set(pair_index):
                    raise ValueError(f"Metric/data samples differ: {directory}")
                counts = np.zeros(args.bins, dtype=np.int64)
                probability_sums = np.zeros(args.bins, dtype=np.float64)
                target_sums = np.zeros(args.bins, dtype=np.float64)
                squared_error_sum = 0.0
                pixel_count = 0
                image_uncertainty = []
                image_dice = []
                for sample in sorted(pair_index):
                    image_path, mask_path = pair_index[sample]
                    probability_path = directory / "probabilities" / f"{sample}.png"
                    if not probability_path.is_file():
                        raise FileNotFoundError(probability_path)
                    probability = np.clip(load_probability(probability_path, args.image_size), 0.0, 1.0)
                    target = resize_mask(mask_path, args.image_size).astype(np.float64)
                    fov = resize_fov(image_path, args.image_size)
                    p = probability[fov]
                    y = target[fov]
                    indices = np.minimum((p * args.bins).astype(int), args.bins - 1)
                    counts += np.bincount(indices, minlength=args.bins)
                    probability_sums += np.bincount(indices, weights=p, minlength=args.bins)
                    target_sums += np.bincount(indices, weights=y, minlength=args.bins)
                    squared_error_sum += float(((p - y) ** 2).sum())
                    pixel_count += len(p)
                    entropy = -(p * np.log2(np.clip(p, 1e-12, 1.0)) + (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1.0)))
                    uncertainty = float(entropy.mean())
                    dice = metric_index[sample]
                    image_uncertainty.append(uncertainty)
                    image_dice.append(dice)
                    per_image_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "sample": sample,
                            "predictive_entropy": uncertainty,
                            "dice": dice,
                            "error": 1.0 - dice,
                        }
                    )
                nonempty = counts > 0
                bin_probability = np.divide(probability_sums, counts, out=np.zeros_like(probability_sums), where=nonempty)
                bin_frequency = np.divide(target_sums, counts, out=np.zeros_like(target_sums), where=nonempty)
                ece = float((counts[nonempty] * np.abs(bin_probability[nonempty] - bin_frequency[nonempty])).sum() / pixel_count)
                brier = float(squared_error_sum / pixel_count)
                uncertainty = np.asarray(image_uncertainty)
                dice = np.asarray(image_dice)
                error = 1.0 - dice
                correlation = spearmanr(uncertainty, error).statistic
                failure_threshold = float(np.quantile(dice, 0.25))
                failure = dice <= failure_threshold
                failure_auroc = binary_auroc(failure, uncertainty)
                order = np.argsort(uncertainty)
                risk_curve = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
                aurc = float(risk_curve.mean())
                run_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "n_images": len(dice),
                        "n_fov_pixels": pixel_count,
                        "foreground_ece": ece,
                        "brier_score": brier,
                        "entropy_error_spearman": float(correlation),
                        "failure_definition": "image Dice <= within-run 25th percentile",
                        "failure_dice_threshold": failure_threshold,
                        "failure_detection_auroc": failure_auroc,
                        "aurc": aurc,
                    }
                )
                for bin_index in range(args.bins):
                    bin_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "bin": bin_index,
                            "lower": bin_index / args.bins,
                            "upper": (bin_index + 1) / args.bins,
                            "count": int(counts[bin_index]),
                            "mean_probability": float(bin_probability[bin_index]),
                            "foreground_frequency": float(bin_frequency[bin_index]),
                        }
                    )

    summary_rows = []
    for dataset in args.datasets:
        for model in args.models:
            subset = [row for row in run_rows if row["dataset"] == dataset and row["model"] == model]
            for metric in ("foreground_ece", "brier_score", "entropy_error_spearman", "failure_detection_auroc", "aurc"):
                values = np.asarray([float(row[metric]) for row in subset])
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "seed_mean": float(np.nanmean(values)),
                        "seed_std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                        "n_seeds": len(values),
                    }
                )
    write_csv(args.output_dir / "calibration_by_run.csv", run_rows)
    write_csv(args.output_dir / "calibration_summary.csv", summary_rows)
    write_csv(args.output_dir / "reliability_bins.csv", bin_rows)
    write_csv(args.output_dir / "image_uncertainty.csv", per_image_rows)
    report = {
        "ece": f"foreground-probability ECE with {args.bins} equal-width bins inside derived FOV",
        "brier": "mean squared foreground-probability error inside derived FOV",
        "uncertainty": "mean binary entropy per image",
        "failure_detection": "AUROC for bottom-quartile image Dice, selected within each frozen run",
        "aurc": "mean cumulative segmentation risk after sorting images by increasing entropy",
        "scope": "descriptive calibration/failure-detection analysis; no threshold or model tuning",
    }
    (args.output_dir / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Calibration analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
