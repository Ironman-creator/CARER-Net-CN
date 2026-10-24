"""从已发布逐图CSV复算组件消融；不载入权重，不进行训练或图像推理。"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = (42, 123, 3407)
ARMS = ("core", "no_bound", "random_init")
ENDPOINTS = ("dice", "thin_centerline_recall")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def calculate(records):
    spec = importlib.util.spec_from_file_location("component_statistics", HERE / "engine/component_summary.py")
    statistics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(statistics)
    indexed = {}
    for arm in ARMS:
        for seed in SEEDS:
            rows = read_csv(records / f"results/{arm}/seed_{seed}/per_image_metrics.csv")
            table = {row["sample"].strip().lower(): row for row in rows}
            if len(rows) != len(table) or len(rows) != 200:
                raise ValueError(f"Expected 200 unique images: {arm}/{seed}")
            indexed[(arm, seed)] = table
    samples = sorted(indexed[("core", 42)])
    if any(sorted(table) != samples for table in indexed.values()):
        raise ValueError("Seed/arm image memberships differ")
    summary, matrices = [], {}
    for arm in ARMS:
        for metric in ENDPOINTS:
            matrix = np.array([[float(indexed[(arm, seed)][sample][metric]) for sample in samples] for seed in SEEDS])
            if not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
                raise ValueError("Invalid metric value")
            matrices[(arm, metric)] = matrix
            summary.append({"arm": arm, "metric": metric, "mean": float(matrix.mean()),
                            "seed_sd": float(matrix.mean(axis=1).std(ddof=1))})
    expected = {(r["baseline"], r["metric"]): r for r in read_csv(records / "paired_component_statistics.csv")}
    rng = np.random.default_rng(20261021)
    paired = []
    for alternative in ("no_bound", "random_init"):
        for metric in ENDPOINTS:
            delta = matrices[("core", metric)] - matrices[(alternative, metric)]
            distribution = statistics.crossed_distribution(delta, 10000, rng)
            values = {"mean_delta": float(delta.mean()),
                "pointwise_95_low": float(np.quantile(distribution, .025)),
                "pointwise_95_high": float(np.quantile(distribution, .975)),
                "bonferroni_98_75_low": float(np.quantile(distribution, .00625)),
                "bonferroni_98_75_high": float(np.quantile(distribution, .99375))}
            recorded = expected[(alternative, metric)]
            for key, value in values.items():
                if abs(value - float(recorded[key])) > 1e-12:
                    raise AssertionError((alternative, metric, key, value, recorded[key]))
            paired.append({"candidate": "core", "baseline": alternative, "metric": metric, **values})
    return summary, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=HERE / "records")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/component_statistics")
    args = parser.parse_args()
    summary, paired = calculate(args.records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("table7_component_means.csv", summary), ("component_paired_intervals.csv", paired)):
        with (args.output_dir / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {"passed": True, "seeds": list(SEEDS), "images_per_run": 200,
        "paired_contrasts_checked": 4, "bootstrap_replicates": 10000,
        "bootstrap_seed": 20261021, "matches_released_statistics": True,
        "model_weights_verified": False, "training_or_inference_started": False,
        "summary": summary, "paired": paired}
    (args.output_dir / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
