"""核对表8七种输入方案的逐图均值、种子标准差及历史配对区间。"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "component_ablation/engine/historical"
SEEDS = (42, 123, 3407)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify():
    source = read_csv(HISTORY / "历史七方案汇总.csv")
    summary = {row["model"]: row for row in read_csv(HISTORY / "raw_color_controls/FIVES_color_controls_test_summary.csv")}
    pairwise = read_csv(HISTORY / "raw_color_controls/FIVES_color_controls_pairwise.csv")
    arrays, findings = {}, []
    core = next(row for row in source if row["model"] == "color_detail_only_unet")
    for arm in source:
        values = []
        for seed in SEEDS:
            path = HISTORY / "raw_per_image" / arm["arm"] / f"seed_{seed}" / "per_image_metrics.csv"
            rows = read_csv(path)
            current = {row["sample"].lower(): float(row["dice"]) for row in rows}
            if len(rows) != 200 or len(current) != 200:
                raise ValueError("Expected 200 unique test images: " + str(path))
            arrays[(arm["arm"], seed)] = current
            values.append(mean(current.values()))
        avg, sd = mean(values), stdev(values)
        for observed, expected in ((avg, float(arm["mean_dice"])), (sd, float(arm["seed_sd_dice"])),
            (avg, float(summary[arm["model"]]["mean_dice"])), (sd, float(summary[arm["model"]]["std_across_seeds_dice"]))):
            if abs(observed - expected) > 1e-12:
                raise ValueError("Input-control mean or SD mismatch: " + arm["arm"])
        findings.append({"arm": arm["arm"], "model": arm["model"], "label": arm["label_zh"],
            "added_parameters": int(arm["added_parameters"]), "mean_dice": avg, "seed_sd_dice": sd})
    reference = set(arrays[(core["arm"], SEEDS[0])])
    for arm, item in zip(source, findings):
        for seed in SEEDS:
            if set(arrays[(arm["arm"], seed)]) != reference:
                raise ValueError("Input-control test samples do not match")
        delta = mean(mean(arrays[(core["arm"], seed)][sample] - arrays[(arm["arm"], seed)][sample]
            for seed in SEEDS) for sample in sorted(reference))
        if arm["arm"] == core["arm"]:
            low = high = None
        else:
            pair = next((row for row in pairwise if {row["candidate"], row["baseline"]} == {core["model"], arm["model"]}), None)
            if pair is None:
                raise ValueError("Historical pairwise comparison missing")
            if pair["candidate"] == core["model"]:
                reported = float(pair["mean_delta_dice"])
                low, high = float(pair["ci95_low_dice"]), float(pair["ci95_high_dice"])
            else:
                reported = -float(pair["mean_delta_dice"])
                low, high = -float(pair["ci95_high_dice"]), -float(pair["ci95_low_dice"])
            if abs(delta - reported) > 1e-12:
                raise ValueError("Paired input-control mean mismatch")
            for observed, field in ((delta, "core_minus_arm_dice"), (low, "core_minus_arm_ci95_low_dice"), (high, "core_minus_arm_ci95_high_dice")):
                if abs(observed - float(arm[field])) > 1e-12:
                    raise ValueError("Historical input-control interval mismatch")
        item.update(core_minus_arm_dice=delta, ci95_low=low, ci95_high=high)
    table = [["方案", "新增参数", "Dice均值±标准差", "Core−该方案（95% CI）"]]
    for row in findings:
        cell = "—" if row["ci95_low"] is None else f'{row["core_minus_arm_dice"]:+.6f} [{row["ci95_low"]:+.6f}, {row["ci95_high"]:+.6f}]'
        table.append([row["label"], str(row["added_parameters"]), f'{row["mean_dice"]:.6f}±{row["seed_sd_dice"]:.6f}', cell])
    expected_path = ROOT / "results/chinese_paper/table_08.csv"
    with expected_path.open(encoding="utf-8-sig", newline="") as handle:
        if list(csv.reader(handle)) != table:
            raise ValueError("Published table 8 differs from recomputed results")
    return {"passed": True, "arms_checked": len(findings), "seeds": list(SEEDS), "images_per_arm_seed": 200,
        "means_sd_and_paired_point_estimates_recomputed": True,
        "intervals": "Historical unadjusted paired-image bootstrap intervals checked against source; not resampled by this entry point.",
        "training_or_inference_started": False, "summary": findings, "table": table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/input_controls_check")
    args = parser.parse_args()
    report = verify()
    if args.output_dir.exists():
        raise FileExistsError("Please use a new output directory: " + str(args.output_dir))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "input_controls_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "table_08.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(report["table"])
    print(json.dumps({key: value for key, value in report.items() if key != "table"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
