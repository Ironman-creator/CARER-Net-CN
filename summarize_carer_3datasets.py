import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASETS = ("DRIVE", "STARE", "CHASE")
SEEDS = (42, 123, 3407)
MODELS = (
    "res_attention_unet_gn",
    "gray_rgb_control_30_unet",
    "color_detail_only_unet",
    "carer_unet",
)
METRICS = (
    "accuracy",
    "sensitivity",
    "specificity",
    "dice",
    "iou",
    "auc_roc",
    "auc_pr",
)
COMPARISONS = (
    ("gray_rgb_control_30_unet", "res_attention_unet_gn"),
    ("color_detail_only_unet", "gray_rgb_control_30_unet"),
    ("carer_unet", "gray_rgb_control_30_unet"),
    ("carer_unet", "color_detail_only_unet"),
    ("carer_unet", "res_attention_unet_gn"),
)
EXPECTED_TEST_COUNTS = {"DRIVE": 20, "STARE": 10, "CHASE": 8}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize frozen CARER experiments on DRIVE, STARE, and CHASE."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gn-run-root", type=Path, required=True)
    parser.add_argument("--carer-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def bootstrap_interval(values, seed, repeats=10000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(values), size=(repeats, len(values)))
    means = values[sampled].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def result_directory(root, dataset, model, seed):
    return root / dataset / model / f"seed_{seed}"


def validation_result_path(args, dataset, model, seed):
    root = args.gn_run_root if model == "res_attention_unet_gn" else args.carer_run_root
    return root / dataset / model / f"seed_{seed}" / "best_result.json"


def main():
    args = parse_args()
    manifest = read_json(args.manifest)
    manifest_records = {
        (record["dataset"], int(record["seed"]), record["model"]): record
        for record in manifest.get("checkpoints", [])
    }
    expected_keys = {
        (dataset, seed, model)
        for dataset in DATASETS
        for seed in SEEDS
        for model in MODELS
    }
    if set(manifest_records) != expected_keys:
        raise RuntimeError("Checkpoint manifest does not contain the expected 36 records")

    metadata = {}
    per_image = {}
    sample_orders = {}
    validation_rows = []
    for dataset in DATASETS:
        expected_count = EXPECTED_TEST_COUNTS[dataset]
        for seed in SEEDS:
            for model in MODELS:
                directory = result_directory(args.result_root, dataset, model, seed)
                metrics_path = directory / "metrics.json"
                rows_path = directory / "per_image_metrics.csv"
                if not metrics_path.is_file() or not rows_path.is_file():
                    raise FileNotFoundError(
                        f"Incomplete test result for {dataset} {model} seed {seed}: {directory}"
                    )
                record = read_json(metrics_path)
                rows = read_csv(rows_path)
                if record.get("dataset") != dataset or record.get("model") != model:
                    raise RuntimeError(f"Metadata mismatch in {metrics_path}")
                if int(record.get("seed")) != seed:
                    raise RuntimeError(f"Seed mismatch in {metrics_path}")
                if int(record.get("test_samples")) != expected_count or len(rows) != expected_count:
                    raise RuntimeError(f"Unexpected test sample count in {directory}")
                if record.get("threshold_source") != "fixed" or abs(
                    float(record.get("threshold")) - 0.5
                ) > 1e-12:
                    raise RuntimeError(f"Threshold protocol mismatch in {metrics_path}")
                if record.get("threshold_selection") is not None:
                    raise RuntimeError(f"Test-set threshold selection detected in {metrics_path}")
                if record.get("fov") != "derived":
                    raise RuntimeError(f"FOV protocol mismatch in {metrics_path}")
                manifest_record = manifest_records[(dataset, seed, model)]
                if record.get("checkpoint_sha256") != manifest_record.get(
                    "checkpoint_sha256"
                ):
                    raise RuntimeError(f"Checkpoint hash mismatch in {metrics_path}")
                order = [row["sample"] for row in rows]
                if dataset not in sample_orders:
                    sample_orders[dataset] = order
                elif order != sample_orders[dataset]:
                    raise RuntimeError(f"Test sample order mismatch in {rows_path}")
                metadata[(dataset, seed, model)] = record
                per_image[(dataset, seed, model)] = {row["sample"]: row for row in rows}

                best_path = validation_result_path(args, dataset, model, seed)
                if not best_path.is_file():
                    raise FileNotFoundError(f"Missing validation result: {best_path}")
                best = read_json(best_path)
                validation_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "best_epoch": best["best_epoch"],
                        "selection_metric": best["selection_metric"],
                        "best_validation_loss": best["best_validation_loss"],
                        "best_validation_dice": best["best_validation_dice"],
                        "best_validation_soft_dice": best["best_validation_soft_dice"],
                    }
                )

    seed_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for model in MODELS:
                record = metadata[(dataset, seed, model)]
                row = {
                    "dataset": dataset,
                    "split": "official_test",
                    "model": model,
                    "seed": seed,
                    "test_images": record["test_samples"],
                    "parameter_count": record["parameter_count"],
                    "threshold": record["threshold"],
                    "fov": record["fov"],
                }
                for metric in METRICS:
                    row[f"mean_{metric}"] = record["metrics"][metric]["mean"]
                    row[f"image_std_{metric}"] = record["metrics"][metric]["std"]
                row["mean_inference_seconds"] = record["metrics"]["inference_seconds"][
                    "mean"
                ]
                seed_rows.append(row)

    model_summary_rows = []
    for dataset in DATASETS:
        for model in MODELS:
            row = {
                "dataset": dataset,
                "split": "official_test",
                "model": model,
                "seeds": "42;123;3407",
                "seed_count": len(SEEDS),
                "test_images_per_seed": EXPECTED_TEST_COUNTS[dataset],
                "parameter_count": metadata[(dataset, 42, model)]["parameter_count"],
            }
            for metric in METRICS:
                values = [
                    float(metadata[(dataset, seed, model)]["metrics"][metric]["mean"])
                    for seed in SEEDS
                ]
                row[f"mean_{metric}"] = float(np.mean(values))
                row[f"seed_std_{metric}"] = sample_std(values)
            timings = [
                float(
                    metadata[(dataset, seed, model)]["metrics"]["inference_seconds"][
                        "mean"
                    ]
                )
                for seed in SEEDS
            ]
            row["mean_inference_seconds"] = float(np.mean(timings))
            row["seed_std_inference_seconds"] = sample_std(timings)
            model_summary_rows.append(row)

    pairwise_seed_rows = []
    per_image_delta_rows = []
    pairwise_summary_rows = []
    dataset_delta_summary = {}
    for dataset_index, dataset in enumerate(DATASETS):
        for comparison_index, (candidate, baseline) in enumerate(COMPARISONS):
            seed_metric_deltas = {metric: [] for metric in METRICS}
            image_seed_deltas = {sample: [] for sample in sample_orders[dataset]}
            for seed in SEEDS:
                candidate_record = metadata[(dataset, seed, candidate)]
                baseline_record = metadata[(dataset, seed, baseline)]
                deltas = {
                    metric: float(candidate_record["metrics"][metric]["mean"])
                    - float(baseline_record["metrics"][metric]["mean"])
                    for metric in METRICS
                }
                for metric in METRICS:
                    seed_metric_deltas[metric].append(deltas[metric])
                seed_dice_deltas = []
                for sample in sample_orders[dataset]:
                    candidate_row = per_image[(dataset, seed, candidate)][sample]
                    baseline_row = per_image[(dataset, seed, baseline)][sample]
                    delta = float(candidate_row["dice"]) - float(baseline_row["dice"])
                    seed_dice_deltas.append(delta)
                    image_seed_deltas[sample].append(delta)
                    per_image_delta_rows.append(
                        {
                            "dataset": dataset,
                            "split": "official_test",
                            "seed": seed,
                            "sample": sample,
                            "baseline": baseline,
                            "candidate": candidate,
                            "baseline_dice": baseline_row["dice"],
                            "candidate_dice": candidate_row["dice"],
                            "delta_dice": delta,
                        }
                    )
                ci_low, ci_high = bootstrap_interval(
                    seed_dice_deltas,
                    seed=20260924 + dataset_index * 100000 + seed * 100 + comparison_index,
                )
                pairwise_seed_rows.append(
                    {
                        "dataset": dataset,
                        "split": "official_test",
                        "seed": seed,
                        "baseline": baseline,
                        "candidate": candidate,
                        **{f"delta_{metric}": deltas[metric] for metric in METRICS},
                        "dice_delta_ci_low": ci_low,
                        "dice_delta_ci_high": ci_high,
                        "images_with_higher_dice": sum(value > 0 for value in seed_dice_deltas),
                        "images_with_equal_dice": sum(value == 0 for value in seed_dice_deltas),
                        "test_images": len(seed_dice_deltas),
                    }
                )

            per_image_mean_deltas = [
                float(np.mean(image_seed_deltas[sample])) for sample in sample_orders[dataset]
            ]
            ci_low, ci_high = bootstrap_interval(
                per_image_mean_deltas,
                seed=20269999 + dataset_index * 100 + comparison_index,
            )
            summary_row = {
                "dataset": dataset,
                "split": "official_test",
                "baseline": baseline,
                "candidate": candidate,
                "seeds": "42;123;3407",
                "test_images": EXPECTED_TEST_COUNTS[dataset],
                "paired_image_bootstrap_dice_ci_low": ci_low,
                "paired_image_bootstrap_dice_ci_high": ci_high,
            }
            for metric in METRICS:
                values = seed_metric_deltas[metric]
                summary_row[f"mean_delta_{metric}"] = float(np.mean(values))
                summary_row[f"seed_std_delta_{metric}"] = sample_std(values)
                summary_row[f"favorable_seeds_{metric}"] = sum(value > 0 for value in values)
            pairwise_summary_rows.append(summary_row)
            dataset_delta_summary[(dataset, candidate, baseline)] = summary_row

    cross_dataset_rows = []
    for candidate, baseline in COMPARISONS:
        row = {
            "baseline": baseline,
            "candidate": candidate,
            "datasets": ";".join(DATASETS),
            "dataset_count": len(DATASETS),
        }
        for metric in METRICS:
            values = [
                float(dataset_delta_summary[(dataset, candidate, baseline)][f"mean_delta_{metric}"])
                for dataset in DATASETS
            ]
            row[f"macro_mean_delta_{metric}"] = float(np.mean(values))
            row[f"dataset_std_delta_{metric}"] = sample_std(values)
            row[f"favorable_datasets_{metric}"] = sum(value > 0 for value in values)
        cross_dataset_rows.append(row)

    full_vs_color = {
        dataset: dataset_delta_summary[
            (dataset, "carer_unet", "color_detail_only_unet")
        ]
        for dataset in DATASETS
    }
    full_vs_control = {
        dataset: dataset_delta_summary[
            (dataset, "carer_unet", "gray_rgb_control_30_unet")
        ]
        for dataset in DATASETS
    }
    report = {
        "protocol": (
            "Frozen FIVES-selected CARER; DRIVE, STARE, and CHASE official test sets; "
            "three fixed seeds; historical explicit train/validation splits; threshold 0.5; "
            "derived FOV; no dataset-specific architecture or hyperparameter tuning."
        ),
        "primary_model": "carer_unet",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "full_carer_minus_color_only_dice": {
            dataset: {
                "mean_delta": row["mean_delta_dice"],
                "bootstrap_ci": [
                    row["paired_image_bootstrap_dice_ci_low"],
                    row["paired_image_bootstrap_dice_ci_high"],
                ],
                "favorable_seeds": row["favorable_seeds_dice"],
            }
            for dataset, row in full_vs_color.items()
        },
        "full_carer_minus_compute_matched_control_dice": {
            dataset: {
                "mean_delta": row["mean_delta_dice"],
                "bootstrap_ci": [
                    row["paired_image_bootstrap_dice_ci_low"],
                    row["paired_image_bootstrap_dice_ci_high"],
                ],
                "favorable_seeds": row["favorable_seeds_dice"],
            }
            for dataset, row in full_vs_control.items()
        },
        "interpretation_rule": (
            "Report all frozen results regardless of direction. Do not tune the architecture, "
            "training schedule, checkpoint choice, or threshold using official test outcomes."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "CARER_3datasets_validation_results.csv", validation_rows)
    write_csv(args.output_dir / "CARER_3datasets_test_seed_results.csv", seed_rows)
    write_csv(args.output_dir / "CARER_3datasets_test_model_summary.csv", model_summary_rows)
    write_csv(args.output_dir / "CARER_3datasets_pairwise_by_seed.csv", pairwise_seed_rows)
    write_csv(args.output_dir / "CARER_3datasets_pairwise_summary.csv", pairwise_summary_rows)
    write_csv(args.output_dir / "CARER_3datasets_cross_dataset_summary.csv", cross_dataset_rows)
    write_csv(
        args.output_dir / "CARER_3datasets_per_image_dice_deltas.csv",
        per_image_delta_rows,
    )
    with (args.output_dir / "CARER_3datasets_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    for row in model_summary_rows:
        print(
            f"{row['dataset']:<6} {row['model']:<30} "
            f"Dice={float(row['mean_dice']):.6f}+/-{float(row['seed_std_dice']):.6f}"
        )
    print(f"Summary written to: {args.output_dir}")


if __name__ == "__main__":
    main()
