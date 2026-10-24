"""核对已发布逐种子/逐图均值，并整理中文论文表2–6；不启动训练。"""
from __future__ import annotations
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE, CONTROL, REFINE = "color_detail_only_unet", "gray_rgb_control_30_unet", "carer_unet"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if not rows:
        raise ValueError(f"Empty output: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_seed_summary(group):
    base = ROOT / "results" / group
    seeds = read(base / "JIIM_stage2_seed_results.csv")
    recorded = read(base / "JIIM_stage2_model_summary.csv")
    groups = defaultdict(list)
    for row in seeds:
        groups[(row["dataset"], row["model"])].append(row)
    result = []
    for row in recorded:
        values = groups[(row["dataset"], row["model"])]
        if sorted(int(value["seed"]) for value in values) != [42, 123, 3407]:
            raise AssertionError("Missing or duplicate training seeds")
        dice = np.array([float(value["dice"]) for value in values])
        mean, sd = float(dice.mean()), float(dice.std(ddof=1))
        if abs(mean-float(row["dice_mean"])) > 1e-12 or abs(sd-float(row["dice_std"])) > 1e-12:
            raise AssertionError((group, row["dataset"], row["model"]))
        result.append({"dataset": row["dataset"], "model": row["model"],
            "dice_mean": mean, "dice_seed_sd": sd, "parameter_count": row.get("parameter_count", ""),
            "zero_dice_seeds": int((dice == 0).sum())})
    return result


def paired_region_table():
    outputs = []
    definitions = (("carer_3datasets", "CARER_3datasets"), ("fives_official_test", "FIVES_official_test"))
    for folder, stem in definitions:
        source = ROOT / "results" / folder
        raw = read(source / f"{stem}_per_image_dice_deltas.csv")
        indexed = defaultdict(list)
        for row in raw:
            indexed[(row["dataset"], row["baseline"], row["candidate"])].append(row)
        for row in read(source / f"{stem}_pairwise_summary.csv"):
            pair = row["baseline"], row["candidate"]
            if pair not in ((CONTROL, CORE), (CORE, REFINE)):
                continue
            records = indexed[(row["dataset"], *pair)]
            identities = [(int(record["seed"]), record["sample"].lower()) for record in records]
            if len(set(identities)) != len(records):
                raise AssertionError("Duplicate seed-image contrast")
            point = float(np.mean([float(record["delta_dice"]) for record in records]))
            if abs(point-float(row["mean_delta_dice"])) > 1e-12:
                raise AssertionError("Paired Dice point estimate differs")
            low, high = row.get("paired_image_bootstrap_dice_ci_low", ""), row.get("paired_image_bootstrap_dice_ci_high", "")
            interval_source = "historical paired-bootstrap output; not recalculated by this script"
            if not low or not high:
                # The original FIVES summary omits interval columns. Retain the
                # displayed Chinese-paper interval, clearly identified as rounded.
                paper = read(ROOT / "results/chinese_paper/table_03.csv")
                paper_row = next(value for value in paper if list(value.values())[0] == row["dataset"])
                displayed = list(paper_row.values())[3 if pair == (CONTROL, CORE) else 4]
                match = re.search(r"\[([^,，]+)[,，]([^\]]+)\]", displayed)
                if not match:
                    raise ValueError(displayed)
                low, high = [value.strip().replace("−", "-") for value in match.groups()]
                interval_source = "rounded displayed Chinese-paper interval; original FIVES aggregate has no CI columns"
            outputs.append({"dataset": row["dataset"], "baseline": pair[0], "candidate": pair[1],
                "mean_delta_dice_recomputed": point,
                "recorded_95_low": low, "recorded_95_high": high,
                "interval_source": interval_source})
    if len(outputs) != 8:
        raise AssertionError("Expected two contrasts for each of four datasets")
    return outputs


def structure_table():
    base = ROOT / "results/fives_structural"
    raw = read(base / "FIVES_逐图结构指标.csv")
    by_model = defaultdict(list)
    for row in raw:
        by_model[row["model"]].append(row)
    recorded = read(base / "FIVES_结构指标配对统计.csv")
    rows = [row for row in recorded if row["candidate"] == CORE and row["baseline"] == CONTROL
        and row["group_type"] == "总体" and row["group"] == "全部"
        and row["analysis_role"] == "exploratory_pointwise" and float(row["ci_level"]) == .95]
    metrics = ("cldice", "skeleton_precision", "skeleton_recall", "branchpoint_f1",
               "thin_centerline_recall", "medium_centerline_recall", "thick_centerline_recall",
               "boundary_f1_euclidean_2px")
    rows = sorted((row for row in rows if row["metric"] in metrics), key=lambda row: metrics.index(row["metric"]))
    if len(rows) != 8:
        raise AssertionError("Expected the eight Chinese-paper structure endpoints")
    result = []
    for row in rows:
        metric = row["metric"]
        if metric not in by_model[CORE][0]:
            continue
        core = float(np.mean([float(value[metric]) for value in by_model[CORE]]))
        control = float(np.mean([float(value[metric]) for value in by_model[CONTROL]]))
        if abs(core-float(row["candidate_mean"])) > 1e-12 or abs(control-float(row["baseline_mean"])) > 1e-12:
            raise AssertionError((metric, "Structure mean differs"))
        result.append({"metric": metric, "core_mean_recomputed": core,
            "control_mean_recomputed": control, "delta_recomputed": core-control,
            "recorded_95_low": row["ci_low"], "recorded_95_high": row["ci_high"],
            "analysis_role": row["analysis_role"]})
    if not any(row["metric"] == "thin_centerline_recall" for row in result):
        raise AssertionError("Thin-centerline endpoint missing")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/chinese_tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"table2_architectures.csv": check_seed_summary("protocol_baselines"),
              "table3_paired_dice.csv": paired_region_table(),
              "table4_structure.csv": structure_table(),
              "table5_external_validation.csv": check_seed_summary("external_validation")}
    benchmark = read(ROOT / "results/efficiency/JIIM_efficiency_benchmark.csv")
    tables["table6_efficiency.csv"] = [{"model": row["model"], "parameters": row["parameters"],
        "macs_G": float(row["macs"])/1e9,
        "peak_memory_MiB": float(row["peak_cuda_allocated_bytes"])/(2**20),
        "latency_ms": 1000*float(row["latency_seconds_mean"]), "fps": row["fps"],
        "gpu": row["gpu"], "warmup": row["warmup_iterations"], "timed_iterations": row["timed_iterations"]}
        for row in benchmark]
    for name, rows in tables.items():
        write(args.output_dir / name, rows)
    report = {"passed": True, "tables": {name: len(rows) for name, rows in tables.items()},
        "seed_means_and_sd_recomputed": True, "paired_point_estimates_recomputed": True,
        "historical_intervals": "Copied from recorded outputs; use the original summarizers on full experiment output to rerun these bootstraps.",
        "component_intervals": "Run component_ablation/analyze_published.py for an independent exact numerical recalculation.",
        "training_or_inference_started": False}
    (args.output_dir / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
