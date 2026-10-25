"""Package existing recorded color controls, without models, images, or training."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


SEEDS = (42, 123, 3407)
ARMS = (
    ("gray", "gray_rgb_control_30_unet", "轮数匹配灰度对照"),
    ("green", "green_channel_unet", "绿色通道"),
    ("clahe", "gray_clahe_unet", "灰度CLAHE"),
    ("projection", "rgb_1x1_projection_unet", "1×1 RGB投影"),
    ("rgb_only", "carer_rgb_only_unet", "仅RGB残差"),
    ("contrast_only", "carer_contrast_only_unet", "仅局部对比残差"),
    ("core", "color_detail_only_unet", "CARER-Core"),
)
RAW_NAMES = (
    "FIVES_color_controls_integrity.json", "FIVES_color_controls_pairwise.csv",
    "FIVES_color_controls_report.txt", "FIVES_color_controls_test_summary.csv",
    "FIVES_color_controls_validation.csv",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing/duplicate CSV columns: {path}")
        result = list(reader)
    if not result:
        raise ValueError(f"Empty CSV: {path}")
    return result


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite historical number")
    return value


def save_csv(path, data):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def copy_record(source, destination, root):
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    if digest(destination) != source_hash:
        raise ValueError(f"Copy hash mismatch: {source}")
    return {"source": str(source.resolve()), "packaged": str(destination.relative_to(root)), "sha256": source_hash}


def core_contrast(pairwise, model, metric):
    core = "color_detail_only_unet"
    if model == core:
        return "", "", "", "not applicable: reference is Core itself"
    matches = [r for r in pairwise if {r["candidate"], r["baseline"]} == {core, model}]
    if len(matches) != 1:
        raise ValueError(f"Expected one Core contrast for {model}, found {len(matches)}")
    row = matches[0]
    if int(row["test_images"]) != 200 or int(row["seeds_averaged_per_image"]) != 3:
        raise ValueError("Historical contrast has an unexpected sample/seed count")
    mean = number(row[f"mean_delta_{metric}"])
    low, high = number(row[f"ci95_low_{metric}"]), number(row[f"ci95_high_{metric}"])
    if low > high:
        raise ValueError("Historical CI bounds are reversed")
    if row["candidate"] != core:
        mean, low, high = -mean, -high, -low
    return mean, low, high, "paired image bootstrap after averaging three seeds; unadjusted 95%"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-per-image", action="store_true", help="Skip optional raw per-image/metrics copies")
    args = parser.parse_args(argv)
    workspace, output = args.workspace_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"complete": False, "created_utc": datetime.now(timezone.utc).isoformat(),
                "role": "historical evidence packaging; not new training, test, or independent replication",
                "workspace_root": str(workspace), "files": [], "optional_missing": [],
                "region_ci": "image bootstrap after three-seed averaging, pointwise unadjusted 95%",
                "structural_ci": "crossed seed x image, post-hoc, four endpoints; 95% and Bonferroni 98.75%"}
    (output / "historical_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source = workspace / "实验数据" / "paper_outputs_fives_color_controls"
    for name in RAW_NAMES:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing required historical artifact: {source / name}")
    raw_summary = rows(source / "FIVES_color_controls_test_summary.csv")
    pairs = rows(source / "FIVES_color_controls_pairwise.csv")
    by_model = {row["model"]: row for row in raw_summary}
    if len(by_model) != len(raw_summary) or set(by_model) != {model for _, model, _ in ARMS}:
        raise ValueError("Historical summary must contain exactly all seven unique arms")
    integrity = json.loads((source / "FIVES_color_controls_integrity.json").read_text(encoding="utf-8-sig"))
    if sorted(int(seed) for seed in integrity["seeds"]) != sorted(SEEDS) or int(integrity["test_samples"]) != 200:
        raise ValueError("Historical integrity metadata does not match the expected three seeds/200 images")
    for name in RAW_NAMES:
        manifest["files"].append(copy_record(source / name, output / "raw_color_controls" / name, output))
    summary_hash = digest(source / "FIVES_color_controls_test_summary.csv")
    pair_hash = digest(source / "FIVES_color_controls_pairwise.csv")
    summary = []
    for arm, model, label in ARMS:
        row = by_model[model]
        if sorted(int(x) for x in row["seeds"].split(";")) != sorted(SEEDS):
            raise ValueError(f"Wrong historical seeds: {model}")
        result = {"arm": arm, "model": model, "label_zh": label,
                  "total_parameters": int(row["parameter_count"]),
                  "added_parameters": int(row["parameter_count"]) - 32778789,
                  "seeds": row["seeds"], "n_test_images": 200,
                  "data_role": "historically observed FIVES official test; not independent replication"}
        for metric in ("dice", "iou", "sensitivity", "specificity"):
            result[f"mean_{metric}"] = number(row[f"mean_{metric}"])
            result[f"seed_sd_{metric}"] = number(row[f"std_across_seeds_{metric}"])
            mean, low, high, definition = core_contrast(pairs, model, metric)
            result[f"core_minus_arm_{metric}"] = mean
            result[f"core_minus_arm_ci95_low_{metric}"] = low
            result[f"core_minus_arm_ci95_high_{metric}"] = high
        result.update({"ci_definition": definition,
                       "summary_source": str((source / "FIVES_color_controls_test_summary.csv").resolve()),
                       "summary_source_sha256": summary_hash,
                       "paired_source": str((source / "FIVES_color_controls_pairwise.csv").resolve()),
                       "paired_source_sha256": pair_hash})
        summary.append(result)
    save_csv(output / "历史七方案汇总.csv", summary)
    if not args.skip_per_image:
        all_keys = None
        arm_seed_means = {}
        for arm, model, _ in ARMS:
            for seed in SEEDS:
                possible_roots = [workspace / "实验数据" / directory / "FIVES" / model / f"seed_{seed}"
                                  for directory in ("result_fives_color_controls", "result_fives_carer_official_test")]
                folder = next((p for p in possible_roots if (p / "per_image_metrics.csv").is_file()), None)
                if folder is None:
                    manifest["optional_missing"].append({"arm": arm, "seed": seed, "artifact": "raw per-image/metrics"})
                    continue
                table = rows(folder / "per_image_metrics.csv")
                keys = [r["sample"].strip().lower() for r in table]
                if len(keys) != 200 or len(set(keys)) != 200:
                    raise ValueError(f"Invalid historical sample keys: {folder}")
                if all_keys is None:
                    all_keys = set(keys)
                elif set(keys) != all_keys:
                    raise ValueError(f"Historical official-test sample mismatch: {folder}")
                raw_mean = sum(number(r["dice"]) for r in table) / len(table)
                arm_seed_means.setdefault(model, []).append(raw_mean)
                per_seed_validation = next((r for r in rows(source / "FIVES_color_controls_validation.csv")
                                            if r["model"] == model and int(r["seed"]) == seed), None)
                for name in ("per_image_metrics.csv", "metrics.json"):
                    original = folder / name
                    if original.is_file():
                        if name == "metrics.json":
                            metadata = json.loads(original.read_text(encoding="utf-8-sig"))
                            if "seed" in metadata and int(metadata["seed"]) != seed:
                                raise ValueError(f"Historical metrics seed mismatch: {original}")
                            if "model" in metadata and metadata["model"] != model:
                                raise ValueError(f"Historical metrics model mismatch: {original}")
                            for field, expected in (("test_samples", 200), ("image_size", 512), ("threshold", 0.5), ("fov", "derived")):
                                if metadata.get(field) != expected:
                                    raise ValueError(f"Historical metrics {field} mismatch: {original}")
                            reported = metadata.get("metrics", {}).get("dice")
                            if isinstance(reported, dict):
                                reported = reported.get("mean")
                            if reported is not None and abs(number(reported) - raw_mean) > 1e-8:
                                raise ValueError(f"Historical metrics/per-image Dice mismatch: {original}")
                        record = copy_record(original, output / "raw_per_image" / arm / f"seed_{seed}" / name, output)
                        record.update(arm=arm, seed=seed, per_image_dice_mean=raw_mean,
                                      validation_record_available=per_seed_validation is not None)
                        manifest["files"].append(record)
                    else:
                        manifest["optional_missing"].append({"arm": arm, "seed": seed, "artifact": name})
        for model, values in arm_seed_means.items():
            if len(values) == len(SEEDS):
                if abs(statistics.mean(values) - number(by_model[model]["mean_dice"])) > 1e-8:
                    raise ValueError(f"Historical summary/raw Dice means mismatch: {model}")
                if abs(statistics.stdev(values) - number(by_model[model]["std_across_seeds_dice"])) > 1e-8:
                    raise ValueError(f"Historical summary/raw Dice SD mismatch: {model}")
    structural_source = workspace / "research_analysis" / "jmbe_projection_strengthening_20261016"
    structural_csv = structural_source / "core_minus_projection_paired_statistics.csv"
    structural_rows = []
    if structural_csv.is_file():
        for name in ("core_minus_projection_paired_statistics.csv", "frozen_protocol.json"):
            if (structural_source / name).is_file():
                manifest["files"].append(copy_record(structural_source / name, output / "raw_projection_structure" / name, output))
        for row in rows(structural_csv):
            if row["metric"] not in {"cldice", "thin_centerline_recall", "branchpoint_f1", "boundary_f1_euclidean_2px"}:
                continue
            structural_rows.append({"metric": row["metric"], "core_minus_projection": number(row["core_minus_projection"]),
                                    "crossed_95_low": number(row["crossed_95_low"]),
                                    "crossed_95_high": number(row["crossed_95_high"]),
                                    "bonferroni_98_75_low": number(row["bonferroni_4_structure_98_75_low"]),
                                    "bonferroni_98_75_high": number(row["bonferroni_4_structure_98_75_high"]),
                                    "analysis_role": "post-hoc historical prediction analysis; not independent replication",
                                    "ci_definition": "crossed training-seed x test-image; four structural endpoints",
                                    "source": str(structural_csv.resolve()), "source_sha256": digest(structural_csv)})
        if len(structural_rows) != 4 or len({r["metric"] for r in structural_rows}) != 4:
            raise ValueError("Expected all four unique historical structural comparisons")
        save_csv(output / "历史Core对投影结构统计.csv", structural_rows)
    else:
        manifest["optional_missing"].append({"artifact": str(structural_csv)})
    lines = ["# 历史局部对比与颜色对照结果", "",
             "本资料汇总已完成的七方案、三种子结果，沿用原训练、测试和阈值设置。FIVES此前已用于方法开发和结果查看，本次属于既有数据上的补充分析。", "",
             "## 全部七方案", "", "|方案|新增参数|Dice均值|种子间SD|Core−该方案Dice|95%图像区间|", "|---|---:|---:|---:|---:|---|"]
    for row in summary:
        interval = "不适用" if row["arm"] == "core" else f"[{row['core_minus_arm_ci95_low_dice']:+.6f}, {row['core_minus_arm_ci95_high_dice']:+.6f}]"
        difference = "—" if row["arm"] == "core" else f"{row['core_minus_arm_dice']:+.6f}"
        lines.append(f"|{row['label_zh']}|{row['added_parameters']}|{row['mean_dice']:.6f}|{row['seed_sd_dice']:.6f}|{difference}|{interval}|")
    lines += ["", "这些区域指标区间先对每幅图的三个种子取平均，再进行配对图像重采样；它们不是种子×图像双向区间，也未做多重比较校正。完整Dice、IoU、敏感度、特异度及每项来源路径与SHA256见历史七方案汇总.csv。", "",
              "## 对照差异与统计结果", "",
              "Core相对RGB-only的差值统一以Core−RGB-only表示，置信区间随差值方向同步变换。Core与1×1 RGB投影的区域Dice差值区间跨零；本次比较未检出明确差异，也未进行等效性检验。",
              "RGB-only有985个新增参数，Core有1201个，两者同时改变了线索与容量。颜色对照与幅度限制、初始化的组件实验分别报告；后者对应中文论文表7。", "",
              "## 后验结构补充", ""]
    if structural_rows:
        lines += ["以下来自冻结预测的后验结构分析；95%与98.75%均为种子×图像双向配对重采样，不使用同文件的image_only列替代。98.75%对应四个结构终点的Bonferroni校正。", "",
                  "|指标|Core−投影|95%双向区间|98.75%双向区间|", "|---|---:|---|---|"]
        for row in structural_rows:
            lines.append(f"|{row['metric']}|{row['core_minus_projection']:+.6f}|[{row['crossed_95_low']:+.6f}, {row['crossed_95_high']:+.6f}]|[{row['bonferroni_98_75_low']:+.6f}, {row['bonferroni_98_75_high']:+.6f}]|")
    else:
        lines.append("未找到可选结构原始统计，未填入或推断任何结构成绩。")
    lines += ["", "## 来源与复核", "", "五份颜色对照原文件按字节复制到raw_color_controls；可找到的逐图CSV与metrics.json原样复制。所有复制文件的原路径、包内路径和SHA256见historical_manifest.json。生成汇总仅作证据整理，CSV保留全精度数值。"]
    (output / "历史局部对比结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in ("历史七方案汇总.csv", "历史局部对比结果.md", "历史Core对投影结构统计.csv"):
        if (output / name).is_file():
            manifest.setdefault("derived_files", []).append({"path": name, "sha256": digest(output / name)})
    manifest["complete"] = True
    (output / "historical_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "arms": 7, "output_dir": str(output), "optional_missing": len(manifest["optional_missing"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
