"""Read-only analysis of completed component runs; never launches model code.

Validation and test reports are deliberately separate. No model, arm, or
checkpoint is selected by this program. A test report requires a training seal.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ARMS = ("core", "no_bound", "random_init")
SEEDS = (42, 123, 3407)
ENDPOINTS = ("dice", "thin_centerline_recall")
TEST_METRICS = ("dice", "iou", "sensitivity", "specificity", "cldice", "thin_centerline_recall")
CONTRASTS = (("core", "no_bound"), ("core", "random_init"))
FAMILY_SIZE = len(ENDPOINTS) * len(CONTRASTS)
NOTICE = (
    "Post-hoc component audit on historically observed FIVES; not independent "
    "replication or a new external cohort. The four-endpoint family contains "
    "two contrasts times two endpoints. No arm selection or testing is triggered."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing or duplicate CSV header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty summary")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def valid_hash(value, label):
    value = str(value).lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid SHA256 in {label}")
    return value


def keyed_metrics(path, expected_count, required, sample_column="sample"):
    rows = read_csv(path)
    index = {}
    for row in rows:
        sample = str(row.get(sample_column, "")).strip().lower()
        if not sample or sample in index:
            raise ValueError(f"Empty or duplicate sample {sample!r}: {path}")
        for metric in required:
            if metric not in row:
                raise ValueError(f"Missing {metric}: {path}")
            number = float(row[metric])
            if not np.isfinite(number) or not 0 <= number <= 1:
                raise ValueError(f"Invalid {metric}={number}: {path}, sample={sample}")
        index[sample] = row
    if len(index) != expected_count:
        raise ValueError(f"Expected {expected_count} samples, found {len(index)}: {path}")
    return index


def verify_training(output_root, phase, expected_validation_images):
    runs, sources, protocol_hashes = {}, [], set()
    for arm in ARMS:
        for seed in SEEDS:
            folder = output_root / "runs" / arm / f"seed_{seed}"
            log_name = "train_log.csv" if (folder / "train_log.csv").is_file() else "training_log.csv"
            paths = {name: folder / name for name in (
                "config.json", "best_result.json", log_name, "validation_per_image.csv"
            )}
            for path in paths.values():
                if not path.is_file():
                    raise ValueError(f"Missing training artifact: {path}")
                sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
            config = read_json(paths["config.json"])
            if config.get("arm") != arm or int(config.get("seed", -1)) != seed:
                raise ValueError(f"Run identity does not match directory: {folder}")
            expected = {"image_size": 512, "epochs": 30}
            for key, value in expected.items():
                if int(config.get(key, -1)) != value:
                    raise ValueError(f"Unexpected {key} in {paths['config.json']}")
            warmup = config.get("warmup", config.get("warmup_epochs"))
            if warmup is None or int(warmup) != 5:
                raise ValueError(f"All three learned arms require five warm-up epochs: {folder}")
            protocol = valid_hash(config.get("protocol_hash", ""), f"{folder}: protocol_hash")
            protocol_hashes.add(protocol)
            initial = valid_hash(config.get("initial_checkpoint_sha256", ""), f"{folder}: initial checkpoint")
            split = valid_hash(config.get("split_sha256", ""), f"{folder}: split")
            log = read_csv(paths[log_name])
            epochs = [int(row.get("epoch", -1)) for row in log]
            if epochs != list(range(31)):
                raise ValueError(f"Training log must contain exactly initial epoch 0 and epochs 1..30 in order: {folder}")
            if log[0].get("phase") != "initial":
                raise ValueError(f"Epoch zero must describe initial validation, not training: {folder}")
            best = read_json(paths["best_result.json"])
            if best.get("completed") is not True or best.get("completed_epochs") != 30:
                raise ValueError(f"Training is not explicitly complete for 30 epochs: {folder}")
            if best.get("selection_metric") != "soft_dice":
                raise ValueError(f"Checkpoints must be selected by validation soft Dice: {folder}")
            for name, expected in (("protocol_hash", protocol), ("initial_checkpoint_sha256", initial),
                                   ("split_sha256", split), ("config_sha256", sha256(paths["config.json"]))):
                if valid_hash(best.get(name, ""), f"{folder}: best_result {name}") != expected:
                    raise ValueError(f"best_result/config {name} mismatch: {folder}")
            if best.get("official_test_accessed") is not False:
                raise ValueError(f"Training result must record official_test_accessed=false: {folder}")
            epoch = best.get("best_epoch", best.get("epoch"))
            if epoch is None or not 0 <= int(epoch) <= 30:
                raise ValueError(f"Invalid validation-selected epoch: {folder}")
            raw_validation = read_csv(paths["validation_per_image.csv"])
            available = raw_validation[0]
            validation_metric = "soft_dice" if "soft_dice" in available else "dice"
            sample_column = "sample" if "sample" in available else "key"
            validation = keyed_metrics(paths["validation_per_image.csv"], expected_validation_images,
                                       (validation_metric,), sample_column=sample_column)
            runs[(arm, seed)] = {
                "config": config, "best": best, "paths": paths, "initial": initial,
                "split": split, "validation": validation, "validation_metric": validation_metric,
            }
    if len(protocol_hashes) != 1:
        raise ValueError("Protocol hashes do not agree across all nine runs")
    for seed in SEEDS:
        reference = runs[("core", seed)]
        for arm in ARMS:
            current = runs[(arm, seed)]
            if current["initial"] != reference["initial"] or current["split"] != reference["split"]:
                raise ValueError(f"Checkpoint/split mismatch within seed {seed}: {arm}")
            if set(current["validation"]) != set(reference["validation"]):
                raise ValueError(f"Validation sample mismatch within seed {seed}: {arm}")
            if current["validation_metric"] != reference["validation_metric"]:
                raise ValueError(f"Validation metric definitions differ within seed {seed}")
    return runs, sources, next(iter(protocol_hashes))


def verify_seal(output_root, runs, protocol_hash):
    path = output_root / "training_seal.json"
    if not path.is_file():
        raise ValueError("Official test summary requires outputs/training_seal.json")
    seal = read_json(path)
    if valid_hash(seal.get("protocol_hash", ""), "training seal") != protocol_hash:
        raise ValueError("Training seal protocol hash mismatch")
    if seal.get("complete") is not True:
        raise ValueError("Training seal must explicitly record complete=true")
    entries = seal.get("runs", seal.get("records"))
    if not isinstance(entries, list):
        raise ValueError("Training seal must contain a runs/records list")
    observed = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each training-seal run must be a JSON object")
        key = (entry.get("arm"), int(entry.get("seed", -1)))
        if key not in runs or key in observed:
            raise ValueError(f"Unknown or duplicate sealed run: {key}")
        observed.add(key)
        for field, name in (("config_sha256", "config.json"), ("best_result_sha256", "best_result.json")):
            if valid_hash(entry.get(field, ""), field) != sha256(runs[key]["paths"][name]):
                raise ValueError(f"Changed training artifact after seal: {key}, {name}")
        checkpoint_relative = entry.get("best_checkpoint")
        if not checkpoint_relative:
            raise ValueError(f"Sealed checkpoint path missing: {key}")
        checkpoint = (output_root / checkpoint_relative).resolve()
        if not checkpoint.is_relative_to(output_root):
            raise ValueError(f"Sealed checkpoint escapes outputs directory: {key}")
        recorded_checkpoint = runs[key]["best"].get("best_checkpoint")
        if (output_root / str(recorded_checkpoint)).resolve() != checkpoint:
            raise ValueError(f"Sealed/best_result checkpoint paths differ: {key}")
        if valid_hash(entry.get("best_checkpoint_sha256", ""), "sealed checkpoint") != sha256(checkpoint):
            raise ValueError(f"Changed checkpoint after seal: {key}")
    if observed != set(runs):
        raise ValueError("Training seal does not include exactly all nine runs")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def crossed_distribution(delta, replicates, rng):
    """Paired arms, independently resampled crossed seed and image indices."""
    n_seeds, n_images = delta.shape
    distribution = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 128):
        count = min(128, replicates - start)
        seed_indices = rng.integers(0, n_seeds, size=(count, n_seeds))
        image_indices = rng.integers(0, n_images, size=(count, n_images))
        sampled = delta[seed_indices[:, :, None], image_indices[:, None, :]]
        distribution[start:start + count] = sampled.mean(axis=(1, 2))
    return distribution


def validation_report(runs):
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            run = runs[(arm, seed)]
            metric = run["validation_metric"]
            values = [float(row[metric]) for row in run["validation"].values()]
            rows.append({"phase": "validation", "arm": arm, "seed": seed,
                         "metric": metric, "n_images": len(values), "mean": float(np.mean(values)),
                         "evaluation_domain": "full input grid; not the official-test FOV metric",
                         "best_epoch": run["best"].get("best_epoch", run["best"].get("epoch", "")),
                         "protocol_hash": run["config"]["protocol_hash"],
                         "per_image_sha256": sha256(run["paths"]["validation_per_image.csv"])})
    return rows


def validate_data_manifest(manifest, expected_count, label):
    """Validate content identities; basenames are only unique within a split."""
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"{label} manifest must list exactly {expected_count} samples")
    keys, image_hashes, pixel_hashes, content_records = set(), set(), set(), []
    for entry in records:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid file record in {label} manifest")
        sample = str(entry.get("sample", "")).strip().lower()
        if not sample or sample in keys:
            raise ValueError(f"Empty/duplicate sample key in {label} manifest")
        keys.add(sample)
        image_hash = valid_hash(entry["image"]["sha256"], f"{label} image")
        mask_hash = valid_hash(entry["mask"]["sha256"], f"{label} mask")
        image_hashes.add(image_hash)
        if "pixel_sha256" in entry["image"]:
            pixel_hashes.add(valid_hash(entry["image"]["pixel_sha256"], f"{label} canonical image pixels"))
        content_records.append({"sample": entry["sample"], "image": image_hash, "mask": mask_hash})
    canonical = json.dumps(content_records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if content_hash != valid_hash(manifest.get("data_sha256", ""), f"{label} data hash"):
        raise ValueError(f"{label} manifest content does not match its data_sha256")
    return keys, image_hashes, pixel_hashes, content_hash


def test_report(output_root, runs, expected_images, replicates, rng_seed):
    data, sources, run_rows = {}, [], []
    manifest_path = output_root / "official_test_data_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Official-test summary requires the post-seal official_test_data_manifest.json")
    manifest = read_json(manifest_path)
    manifest_hash = sha256(manifest_path)
    manifest_keys, test_image_hashes, test_pixel_hashes, data_hash = validate_data_manifest(
        manifest, expected_images, "Official test")
    training_manifest_path = output_root / "training_data_manifest.json"
    if not training_manifest_path.is_file():
        raise ValueError("Official-test summary requires the 600-image training_data_manifest.json")
    training_manifest = read_json(training_manifest_path)
    _, training_image_hashes, training_pixel_hashes, training_data_hash = validate_data_manifest(
        training_manifest, 600, "Training pool")
    if training_image_hashes & test_image_hashes:
        raise ValueError("Training/test leakage: identical image file content SHA256 across splits")
    if training_pixel_hashes and test_pixel_hashes and training_pixel_hashes & test_pixel_hashes:
        raise ValueError("Training/test leakage: identical canonical image pixel SHA256 across splits")
    sources.append({"path": str(manifest_path.resolve()), "sha256": manifest_hash,
                    "test_data_sha256": data_hash})
    sources.append({"path": str(training_manifest_path.resolve()), "sha256": sha256(training_manifest_path),
                    "training_data_sha256": training_data_hash})
    test_keys = manifest_keys
    for arm in ARMS:
        for seed in SEEDS:
            folder = output_root / "results" / arm / f"seed_{seed}"
            metrics_path, per_image_path = folder / "metrics.json", folder / "per_image_metrics.csv"
            for path in (metrics_path, per_image_path):
                if not path.is_file():
                    raise ValueError(f"Missing official-test artifact: {path}")
                sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
            aggregate = read_json(metrics_path)
            if valid_hash(aggregate.get("test_data_sha256", ""), "test metrics data hash") != data_hash:
                raise ValueError(f"Different official-test image/mask content was used: {folder}")
            if valid_hash(aggregate.get("test_data_manifest_sha256", ""), "test metrics data-manifest hash") != manifest_hash:
                raise ValueError(f"Official-test manifest hash differs: {folder}")
            for key, expected in (("arm", arm), ("seed", seed), ("image_size", 512), ("threshold", 0.5)):
                if key in aggregate and aggregate[key] != expected:
                    raise ValueError(f"Unexpected test {key} in {metrics_path}")
            if aggregate.get("test_samples") != expected_images:
                raise ValueError(f"Official-test metrics must record {expected_images} test samples: {folder}")
            training = runs[(arm, seed)]
            for field, expected in (("config_sha256", sha256(training["paths"]["config.json"])),
                                    ("protocol_hash", training["config"]["protocol_hash"].lower()),
                                    ("initial_checkpoint_sha256", training["initial"]),
                                    ("split_sha256", training["split"])):
                if valid_hash(aggregate.get(field, ""), f"{folder}: {field}") != expected:
                    raise ValueError(f"Test/training {field} mismatch: {folder}")
            checkpoint = (output_root / training["best"]["best_checkpoint"]).resolve()
            if valid_hash(aggregate.get("best_checkpoint_sha256", ""), "test checkpoint") != sha256(checkpoint):
                raise ValueError(f"Test checkpoint SHA256 does not match sealed checkpoint: {folder}")
            table = keyed_metrics(per_image_path, expected_images, TEST_METRICS)
            if set(table) != test_keys:
                raise ValueError(f"Test sample keys differ from the frozen official-test manifest: {per_image_path}")
            data[(arm, seed)] = table
            for endpoint in TEST_METRICS:
                run_rows.append({"phase": "official_test", "arm": arm, "seed": seed,
                                 "metric": endpoint, "n_images": len(table),
                                 "mean": float(np.mean([float(row[endpoint]) for row in table.values()])),
                                 "per_image_sha256": sha256(per_image_path)})
    ordered = sorted(test_keys)
    rng = np.random.default_rng(rng_seed)
    contrasts = []
    for candidate, baseline in CONTRASTS:
        for endpoint in ENDPOINTS:
            a = np.array([[float(data[(candidate, seed)][sample][endpoint]) for sample in ordered] for seed in SEEDS])
            b = np.array([[float(data[(baseline, seed)][sample][endpoint]) for sample in ordered] for seed in SEEDS])
            delta = a - b
            distribution = crossed_distribution(delta, replicates, rng)
            low, high = np.quantile(distribution, [0.025, 0.975])
            adjusted_low, adjusted_high = np.quantile(distribution, [0.05 / (2 * FAMILY_SIZE), 1 - 0.05 / (2 * FAMILY_SIZE)])
            row = {"candidate": candidate, "baseline": baseline, "metric": endpoint,
                   "n_seeds": len(SEEDS), "n_images": len(ordered),
                   "candidate_mean": float(a.mean()), "baseline_mean": float(b.mean()),
                   "mean_delta": float(delta.mean()), "pointwise_95_low": float(low),
                   "pointwise_95_high": float(high), "family_size": FAMILY_SIZE,
                   "bonferroni_ci_level": 0.9875, "bonferroni_98_75_low": float(adjusted_low),
                   "bonferroni_98_75_high": float(adjusted_high),
                   "adjusted_direction": "positive" if adjusted_low > 0 else "negative" if adjusted_high < 0 else "inconclusive",
                   "positive_image_fraction_after_seed_mean": float((delta.mean(axis=0) > 0).mean()),
                   "resampling": "crossed training-seed and test-image; paired arms",
                   "bootstrap_replicates": replicates, "bootstrap_seed": rng_seed,
                   "analysis_role": "post-hoc component audit; historically observed FIVES"}
            row.update({f"seed_{seed}_delta": float(delta[i].mean()) for i, seed in enumerate(SEEDS)})
            contrasts.append(row)
    return run_rows, contrasts, sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--expected-validation-images", type=int, default=60)
    parser.add_argument("--expected-test-images", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261021)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    destination = output_root / "summaries" / args.phase
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "status.json"
    status = {"phase": args.phase, "complete": False, "notice": NOTICE,
              "created_utc": datetime.now(timezone.utc).isoformat()}
    try:
        if args.bootstrap < 10000:
            raise ValueError("At least 10000 bootstrap replicates are required")
        runs, sources, protocol_hash = verify_training(output_root, args.phase, args.expected_validation_images)
        if args.phase == "validation":
            rows = validation_report(runs)
            write_csv(destination / "validation_run_summary.csv", rows)
            message = "验证集结果仅描述各运行，不选择方案，不读取或触发正式测试。"
        else:
            sources.append(verify_seal(output_root, runs, protocol_hash))
            rows, contrasts, test_sources = test_report(output_root, runs, args.expected_test_images, args.bootstrap, args.bootstrap_seed)
            sources.extend(test_sources)
            write_csv(destination / "official_test_run_summary.csv", rows)
            write_csv(destination / "paired_component_statistics.csv", contrasts)
            lines = ["# FIVES 组件后验审计", "", "本实验使用历史上已查看的 FIVES，不是独立重复验证，也不是新增外部队列。",
                     "仅汇总全部三方案、三种子；不选择方案，不运行训练或测试。",
                     "两个比较乘以 Dice、细血管中心线召回两个终点，共四项；同时报告点态95%与Bonferroni98.75%区间。", "",
                     "|比较|终点|差值|95%双向区间|98.75%双向区间|", "|---|---|---:|---|---|"]
            for row in contrasts:
                lines.append(f"|{row['candidate']}−{row['baseline']}|{row['metric']}|{row['mean_delta']:+.6f}|[{row['pointwise_95_low']:+.6f}, {row['pointwise_95_high']:+.6f}]|[{row['bonferroni_98_75_low']:+.6f}, {row['bonferroni_98_75_high']:+.6f}]|")
            lines += ["", "区间跨零不证明两方案等效；细血管召回不替代误检、区域重叠与时延评价。", "逐图键、配置、初始检查点、划分与训练封存记录的哈希见 status.json。"]
            (destination / "组件审计报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            message = "全部三方案九运行及封存记录已核验，生成配对统计。"
        status.update(complete=True, protocol_hash=protocol_hash, sources=sources, message=message)
        write_json(status_path, status)
        print(json.dumps({"complete": True, "phase": args.phase, "directory": str(destination), "message": message}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        status["error"] = str(error)
        write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
