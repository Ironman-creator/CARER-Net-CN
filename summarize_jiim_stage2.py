from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "sensitivity", "specificity", "dice", "iou", "auc_roc", "auc_pr")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260927)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_delta(delta, count, rng):
    delta = np.asarray(delta, dtype=np.float64)
    means = np.empty(count, dtype=np.float64)
    for start in range(0, count, 1000):
        current = min(1000, count - start)
        indices = rng.integers(0, len(delta), size=(current, len(delta)))
        means[start : start + current] = delta[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    seed_rows = []
    image_values = defaultdict(list)
    sample_sets = {}

    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                directory = args.result_root / dataset / model / f"seed_{seed}"
                metrics_path = directory / "metrics.json"
                image_path = directory / "per_image_metrics.csv"
                if not metrics_path.is_file() or not image_path.is_file():
                    raise FileNotFoundError(f"Incomplete result directory: {directory}")
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                seed_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        **{metric: payload["metrics"][metric]["mean"] for metric in METRICS},
                        "inference_seconds": payload["metrics"]["inference_seconds"]["mean"],
                        "parameter_count": payload["parameter_count"],
                    }
                )
                image_rows = read_csv(image_path)
                samples = [row["sample"].lower() for row in image_rows]
                if len(samples) != len(set(samples)):
                    raise ValueError(f"Duplicate samples in {image_path}")
                sample_sets[(dataset, model, seed)] = set(samples)
                for row in image_rows:
                    sample = row["sample"].lower()
                    for metric in METRICS:
                        image_values[(dataset, model, sample, metric)].append(float(row[metric]))

    for dataset in args.datasets:
        reference = sample_sets[(dataset, args.models[0], args.seeds[0])]
        for model in args.models:
            for seed in args.seeds:
                observed = sample_sets[(dataset, model, seed)]
                if observed != reference:
                    missing = sorted(reference - observed)
                    extra = sorted(observed - reference)
                    raise ValueError(
                        f"Sample mismatch for {dataset}/{model}/seed_{seed}: "
                        f"missing={missing[:10]}, extra={extra[:10]}"
                    )
        for model in args.models:
            for sample in reference:
                for metric in METRICS:
                    values = image_values[(dataset, model, sample, metric)]
                    if len(values) != len(args.seeds):
                        raise ValueError(
                            f"Expected {len(args.seeds)} seed values for "
                            f"{dataset}/{model}/{sample}/{metric}, found {len(values)}"
                        )

    model_rows = []
    for dataset in args.datasets:
        for model in args.models:
            subset = [row for row in seed_rows if row["dataset"] == dataset and row["model"] == model]
            row = {"dataset": dataset, "model": model, "seeds": len(subset)}
            for metric in (*METRICS, "inference_seconds"):
                values = np.asarray([float(item[metric]) for item in subset])
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row["parameter_count"] = int(subset[0]["parameter_count"])
            model_rows.append(row)

    pairwise_rows = []
    comparisons = []
    if "res_attention_unet_gn" in args.models:
        comparisons.extend(
            (model, "res_attention_unet_gn")
            for model in args.models
            if model != "res_attention_unet_gn"
        )
    if "gray_rgb_control_30_unet" in args.models:
        comparisons.extend(
            (model, "gray_rgb_control_30_unet")
            for model in args.models
            if model != "gray_rgb_control_30_unet"
            if model in args.models
        )
    if "color_detail_only_unet" in args.models and "carer_unet" in args.models:
        comparisons.append(("carer_unet", "color_detail_only_unet"))
    if "carer_nonnegative_unet" in args.models and "carer_unet" in args.models:
        comparisons.append(("carer_nonnegative_unet", "carer_unet"))
    if "carer_unet" in args.models:
        comparisons.extend(
            ("carer_unet", baseline)
            for baseline in args.models
            if baseline not in {
                "carer_unet",
                "color_detail_only_unet",
                "gray_rgb_control_30_unet",
            }
        )
    comparisons = list(dict.fromkeys(comparisons))

    for dataset in args.datasets:
        samples = sorted(
            {
                key[2]
                for key in image_values
                if key[0] == dataset and key[1] == args.models[0]
            }
        )
        for candidate, baseline in comparisons:
            for metric in METRICS:
                candidate_values = np.asarray(
                    [np.mean(image_values[(dataset, candidate, sample, metric)]) for sample in samples]
                )
                baseline_values = np.asarray(
                    [np.mean(image_values[(dataset, baseline, sample, metric)]) for sample in samples]
                )
                delta = candidate_values - baseline_values
                low, high = bootstrap_delta(delta, args.bootstrap, rng)
                pairwise_rows.append(
                    {
                        "dataset": dataset,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "n_images": len(samples),
                        "candidate_mean": float(candidate_values.mean()),
                        "baseline_mean": float(baseline_values.mean()),
                        "mean_delta": float(delta.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "positive_fraction": float(np.mean(delta > 0)),
                    }
                )

    write_csv(args.output_dir / "JIIM_stage2_seed_results.csv", seed_rows)
    write_csv(args.output_dir / "JIIM_stage2_model_summary.csv", model_rows)
    if pairwise_rows:
        write_csv(args.output_dir / "JIIM_stage2_pairwise_bootstrap.csv", pairwise_rows)
    report = {
        "datasets": args.datasets,
        "models": args.models,
        "seeds": args.seeds,
        "statistical_unit": "test image after averaging seeds",
        "inference_scope": "conditional on the supplied trained seeds; no patient clustering or algorithm-population p values",
        "intervals": "paired image bootstrap confidence intervals; exploratory unless separately pre-specified",
        "bootstrap": args.bootstrap,
        "result_root": str(args.result_root.resolve()),
    }
    (args.output_dir / "JIIM_stage2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"JIIM stage-2 summary written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
