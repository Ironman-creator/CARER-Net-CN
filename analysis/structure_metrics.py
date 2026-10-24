"""Audited stage-1 analysis for frozen FIVES official-test predictions.

This revision deliberately separates two levels of evidence:

* Two pre-specified primary endpoints (overall Dice and overall clDice for
  CARER versus the compute-matched control) use Bonferroni-compatible 97.5%
  two-way bootstrap confidence intervals.
* Disease, quality, calibre, junction and boundary analyses are exploratory
  and use pointwise 95% intervals.  No rank-test p values are mixed with mean
  bootstrap intervals.

Training seed and test image are crossed sources of variation.  Each bootstrap
replicate therefore resamples both the three seeds and the selected images.
FIVES subject identifiers are not available in the local release, so the
result remains image-level conditional inference and cannot account for two
eyes from the same person.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


MODELS = (
    "res_attention_unet_gn",
    "gray_rgb_control_30_unet",
    "color_detail_only_unet",
    "carer_unet",
)
SEEDS = (42, 123, 3407)
COMPARISONS = (
    ("carer_unet", "gray_rgb_control_30_unet", "CARER-计算匹配对照"),
    ("carer_unet", "color_detail_only_unet", "CARER-颜色模块"),
    ("color_detail_only_unet", "gray_rgb_control_30_unet", "颜色模块-计算匹配对照"),
)
DISEASE_NAMES = {"A": "AMD", "D": "DR", "G": "青光眼", "N": "正常"}
PRIMARY_KEYS = {
    ("CARER-计算匹配对照", "dice", "总体", "全部"),
    ("CARER-计算匹配对照", "cldice", "总体", "全部"),
}
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--fives-root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260927)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--boundary-tolerance", type=float, default=2.0)
    parser.add_argument("--junction-tolerance", type=float, default=2.0)
    parser.add_argument("--minimum-component-pixels", type=int, default=10)
    parser.add_argument("--structure-workers", type=int, default=min(6, os.cpu_count() or 1))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def aggregate_hash(paths: list[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item).lower()):
        digest.update(str(path.relative_to(base)).replace("\\", "/").encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest().upper()


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        sample = row["sample"].strip().lower()
        disease = row["Disease"].strip().upper()
        if disease not in DISEASE_NAMES:
            raise ValueError(f"Unknown disease code for {sample}: {disease}")
        number = int(row["Number"])
        if sample != f"{number}_{disease.lower()}":
            raise ValueError(f"Metadata sample/number/disease mismatch: {row}")
        for field in ("IC", "Blur", "LC"):
            if int(row[field]) not in (0, 1):
                raise ValueError(f"{field} must be 0/1 for {sample}")
        if sample in metadata:
            raise ValueError(f"Duplicate metadata sample: {sample}")
        metadata[sample] = {
            **row,
            "sample": sample,
            "Disease": disease,
            "Number": str(number),
        }
    if len(metadata) != 200:
        raise ValueError(f"Expected 200 metadata rows, found {len(metadata)}")
    counts = Counter(row["Disease"] for row in metadata.values())
    if counts != Counter({"A": 50, "D": 50, "G": 50, "N": 50}):
        raise ValueError(f"Unexpected FIVES disease counts: {dict(counts)}")
    return metadata


def metric_rows(result_root: Path, metadata: dict[str, dict[str, str]]) -> list[dict]:
    rows: list[dict] = []
    expected_samples = set(metadata)
    for model in MODELS:
        for seed in SEEDS:
            path = result_root / model / f"seed_{seed}" / "per_image_metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            raw_rows = read_csv(path)
            samples = [row["sample"].lower() for row in raw_rows]
            if len(samples) != 200 or len(set(samples)) != 200 or set(samples) != expected_samples:
                raise ValueError(f"Incomplete/mismatched samples in {path}")
            for raw in raw_rows:
                sample = raw["sample"].lower()
                values = {
                    key: float(raw[key])
                    for key in (
                        "accuracy",
                        "sensitivity",
                        "specificity",
                        "dice",
                        "iou",
                        "auc_roc",
                        "auc_pr",
                    )
                }
                if not all(np.isfinite(list(values.values()))):
                    raise ValueError(f"Non-finite metric in {path}: {sample}")
                rows.append(
                    {
                        "sample": sample,
                        "model": model,
                        "seed": seed,
                        "disease": metadata[sample]["Disease"],
                        "IC": int(metadata[sample]["IC"]),
                        "Blur": int(metadata[sample]["Blur"]),
                        "LC": int(metadata[sample]["LC"]),
                        **values,
                    }
                )
    return rows


def subgroup_specs(metadata: dict[str, dict[str, str]]) -> list[tuple[str, str, set[str]]]:
    specs: list[tuple[str, str, set[str]]] = [("总体", "全部", set(metadata))]
    for code, name in DISEASE_NAMES.items():
        specs.append(("病种", name, {s for s, row in metadata.items() if row["Disease"] == code}))
    for field, label in (("IC", "照明/颜色"), ("Blur", "模糊"), ("LC", "低对比")):
        specs.append((label, "质量良好", {s for s, row in metadata.items() if int(row[field]) == 1}))
        specs.append((label, "质量较差", {s for s, row in metadata.items() if int(row[field]) == 0}))
    return specs


def value_index(rows: list[dict], metric: str) -> dict[tuple[str, int, str], float]:
    index: dict[tuple[str, int, str], float] = {}
    for row in rows:
        key = (row["model"], int(row["seed"]), row["sample"])
        if key in index:
            raise ValueError(f"Duplicate metric row: {key}")
        index[key] = float(row[metric])
    return index


def paired_matrices(
    index: dict[tuple[str, int, str], float],
    candidate: str,
    baseline: str,
    samples: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ordered = sorted(samples)
    candidate_values = np.asarray(
        [[index[(candidate, seed, sample)] for sample in ordered] for seed in SEEDS],
        dtype=np.float64,
    )
    baseline_values = np.asarray(
        [[index[(baseline, seed, sample)] for sample in ordered] for seed in SEEDS],
        dtype=np.float64,
    )
    return candidate_values, baseline_values, ordered


def crossed_bootstrap_distribution(
    delta: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    if delta.ndim != 2 or delta.shape[0] != len(SEEDS) or delta.shape[1] == 0:
        raise ValueError(f"Expected seed x image delta matrix, got {delta.shape}")
    distribution = np.empty(count, dtype=np.float64)
    chunk = 500
    for start in range(0, count, chunk):
        current = min(chunk, count - start)
        seed_indices = rng.integers(
            0, delta.shape[0], size=(current, delta.shape[0])
        )
        image_indices = rng.integers(
            0, delta.shape[1], size=(current, delta.shape[1])
        )
        sampled = delta[
            seed_indices[:, :, None], image_indices[:, None, :]
        ]
        distribution[start : start + current] = sampled.mean(axis=(1, 2))
    return distribution


def paired_stats(
    candidate: np.ndarray,
    baseline: np.ndarray,
    bootstrap: int,
    rng: np.random.Generator,
    confidence: float,
) -> dict:
    delta = candidate - baseline
    distribution = crossed_bootstrap_distribution(delta, bootstrap, rng)
    alpha = 1.0 - confidence
    low, high = np.quantile(distribution, [alpha / 2.0, 1.0 - alpha / 2.0])
    seed_deltas = delta.mean(axis=1)
    if low > 0:
        evidence = "positive"
    elif high < 0:
        evidence = "negative"
    else:
        evidence = "inconclusive"
    return {
        "n_images": int(delta.shape[1]),
        "n_seeds": int(delta.shape[0]),
        "candidate_mean": float(candidate.mean()),
        "baseline_mean": float(baseline.mean()),
        "mean_delta": float(delta.mean()),
        "median_image_delta_after_seed_mean": float(np.median(delta.mean(axis=0))),
        "ci_level": confidence,
        "ci_low": float(low),
        "ci_high": float(high),
        "evidence": evidence,
        "positive_image_fraction_after_seed_mean": float(np.mean(delta.mean(axis=0) > 0)),
        "seed_42_delta": float(seed_deltas[0]),
        "seed_123_delta": float(seed_deltas[1]),
        "seed_3407_delta": float(seed_deltas[2]),
    }


def analysis_role(comparison: str, metric: str, group_type: str, group: str) -> tuple[str, float]:
    if (comparison, metric, group_type, group) in PRIMARY_KEYS:
        return "primary_bonferroni_family_of_2", 0.975
    return "exploratory_pointwise", 0.95


def paired_analysis(
    rows: list[dict],
    metadata: dict[str, dict[str, str]],
    metrics: tuple[str, ...],
    bootstrap: int,
    rng: np.random.Generator,
) -> list[dict]:
    output: list[dict] = []
    specs = subgroup_specs(metadata)
    for metric in metrics:
        index = value_index(rows, metric)
        for candidate, baseline, comparison in COMPARISONS:
            for group_type, group, samples in specs:
                role, confidence = analysis_role(comparison, metric, group_type, group)
                candidate_values, baseline_values, _ = paired_matrices(
                    index, candidate, baseline, samples
                )
                output.append(
                    {
                        "comparison": comparison,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "group_type": group_type,
                        "group": group,
                        "analysis_role": role,
                        **paired_stats(
                            candidate_values,
                            baseline_values,
                            bootstrap,
                            rng,
                            confidence,
                        ),
                    }
                )
    return output


def disease_adjusted_quality_interactions(
    rows: list[dict],
    metadata: dict[str, dict[str, str]],
    bootstrap: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Estimate method-by-quality interaction adjusted for disease category.

    The coefficient is the CARER-control Dice gain in poor-quality images minus
    that in good-quality images after adding disease indicators.  It is not a
    causal quality effect and all rows are explicitly exploratory.
    """

    index = value_index(rows, "dice")
    candidate, baseline = "carer_unet", "gray_rgb_control_30_unet"
    samples = sorted(metadata)
    delta = np.asarray(
        [
            [index[(candidate, seed, sample)] - index[(baseline, seed, sample)] for sample in samples]
            for seed in SEEDS
        ],
        dtype=np.float64,
    )
    disease = np.asarray([metadata[sample]["Disease"] for sample in samples])
    output = []
    for field, label in (("IC", "照明/颜色"), ("Blur", "模糊"), ("LC", "低对比")):
        poor = np.asarray([1 - int(metadata[sample][field]) for sample in samples], dtype=np.float64)
        design = np.column_stack(
            [
                np.ones(len(samples)),
                poor,
                (disease == "D").astype(float),
                (disease == "G").astype(float),
                (disease == "N").astype(float),
            ]
        )
        averaged = delta.mean(axis=0)
        coefficient = float((np.linalg.pinv(design) @ averaged)[1])
        distribution = np.empty(bootstrap, dtype=np.float64)
        for replicate in range(bootstrap):
            seed_indices = rng.integers(0, len(SEEDS), size=len(SEEDS))
            image_indices = rng.integers(0, len(samples), size=len(samples))
            y = delta[seed_indices].mean(axis=0)[image_indices]
            x = design[image_indices]
            distribution[replicate] = float((np.linalg.pinv(x) @ y)[1])
        low, high = np.quantile(distribution, [0.025, 0.975])
        poor_counts = Counter(disease[poor == 1])
        output.append(
            {
                "quality_field": field,
                "quality_label": label,
                "metric": "dice",
                "interpretation": "adjusted poor-minus-good difference in CARER-control gain",
                "n_poor": int(poor.sum()),
                "n_good": int((1 - poor).sum()),
                "poor_disease_counts": ";".join(
                    f"{code}:{poor_counts.get(code, 0)}" for code in DISEASE_NAMES
                ),
                "adjusted_interaction": coefficient,
                "ci_level": 0.95,
                "ci_low": float(low),
                "ci_high": float(high),
                "evidence": "positive" if low > 0 else "negative" if high < 0 else "inconclusive",
                "analysis_role": "exploratory_pointwise",
                "warning": (
                    "LC poor is sparse and disease-confounded; interpret descriptively"
                    if field == "LC"
                    else "observational subgroup interaction; not a causal quality effect"
                ),
            }
        )
    return output


def resize_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("L").resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 127


def resize_rgb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def skeletonize_zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Standard two-subiteration Zhang-Suen one-pixel thinning."""

    image = np.asarray(mask, dtype=np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p1 = padded[1:-1, 1:-1]
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbours)
            transitions = sum(
                ((neighbours[index] == 0) & (neighbours[(index + 1) % 8] == 1))
                for index in range(8)
            )
            if step == 0:
                constraint_a = p2 * p4 * p6 == 0
                constraint_b = p4 * p6 * p8 == 0
            else:
                constraint_a = p2 * p4 * p8 == 0
                constraint_b = p2 * p6 * p8 == 0
            remove = (
                (p1 == 1)
                & (count >= 2)
                & (count <= 6)
                & (transitions == 1)
                & constraint_a
                & constraint_b
            )
            if remove.any():
                image[remove] = 0
                changed = True
    return image.astype(bool)


def branch_nodes(skeleton: np.ndarray) -> np.ndarray:
    neighbourhood = ndi.convolve(
        skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
    ) - skeleton.astype(np.uint8)
    junction_pixels = skeleton & (neighbourhood >= 3)
    labels, count = ndi.label(junction_pixels, structure=np.ones((3, 3), dtype=np.uint8))
    nodes: list[tuple[float, float]] = []
    for label in range(1, count + 1):
        coordinates = np.argwhere(labels == label)
        centre = coordinates.mean(axis=0)
        representative = coordinates[np.argmin(((coordinates - centre) ** 2).sum(axis=1))]
        nodes.append((float(representative[0]), float(representative[1])))
    return np.asarray(nodes, dtype=np.float64).reshape((-1, 2))


def match_nodes(predicted: np.ndarray, truth: np.ndarray, tolerance: float) -> int:
    if len(predicted) == 0 or len(truth) == 0:
        return 0
    distances = np.sqrt(((predicted[:, None, :] - truth[None, :, :]) ** 2).sum(axis=2))
    pairs = np.argwhere(distances <= tolerance)
    ordered = sorted(
        ((float(distances[pred, gt]), int(pred), int(gt)) for pred, gt in pairs),
        key=lambda item: item[0],
    )
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    for _, predicted_index, truth_index in ordered:
        if predicted_index not in used_predicted and truth_index not in used_truth:
            used_predicted.add(predicted_index)
            used_truth.add(truth_index)
    return len(used_truth)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + EPS))


def filtered_component_count(mask: np.ndarray, minimum_pixels: int) -> tuple[int, float]:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    retained = sizes[sizes >= minimum_pixels]
    largest_fraction = safe_ratio(float(sizes.max()), float(mask.sum()))
    return int(len(retained)), largest_fraction


def structural_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    fov: np.ndarray,
    truth_skeleton: np.ndarray,
    truth_radius: np.ndarray,
    boundary_tolerance: float,
    junction_tolerance: float,
    minimum_component_pixels: int,
) -> dict[str, float]:
    prediction = prediction & fov
    truth = truth & fov
    predicted_skeleton = skeletonize_zhang_suen(prediction)

    skeleton_precision = safe_ratio((predicted_skeleton & truth).sum(), predicted_skeleton.sum())
    skeleton_recall = safe_ratio((truth_skeleton & prediction).sum(), truth_skeleton.sum())
    cldice = safe_ratio(2 * skeleton_precision * skeleton_recall, skeleton_precision + skeleton_recall)

    predicted_nodes = branch_nodes(predicted_skeleton)
    truth_nodes = branch_nodes(truth_skeleton)
    matched_nodes = match_nodes(predicted_nodes, truth_nodes, junction_tolerance)
    branch_precision = safe_ratio(matched_nodes, len(predicted_nodes))
    branch_recall = safe_ratio(matched_nodes, len(truth_nodes))
    branch_f1 = safe_ratio(2 * branch_precision * branch_recall, branch_precision + branch_recall)

    thin_centreline = truth_skeleton & (truth_radius <= 1.5)
    medium_centreline = truth_skeleton & (truth_radius > 1.5) & (truth_radius <= 3.0)
    thick_centreline = truth_skeleton & (truth_radius > 3.0)
    thin_recall = safe_ratio((prediction & thin_centreline).sum(), thin_centreline.sum())
    medium_recall = safe_ratio((prediction & medium_centreline).sum(), medium_centreline.sum())
    thick_recall = safe_ratio((prediction & thick_centreline).sum(), thick_centreline.sum())

    truth_boundary = truth & ~ndi.binary_erosion(truth)
    predicted_boundary = prediction & ~ndi.binary_erosion(prediction)
    distance_to_truth_boundary = ndi.distance_transform_edt(~truth_boundary)
    distance_to_predicted_boundary = ndi.distance_transform_edt(~predicted_boundary)
    boundary_precision = safe_ratio(
        (predicted_boundary & (distance_to_truth_boundary <= boundary_tolerance)).sum(),
        predicted_boundary.sum(),
    )
    boundary_recall = safe_ratio(
        (truth_boundary & (distance_to_predicted_boundary <= boundary_tolerance)).sum(),
        truth_boundary.sum(),
    )
    boundary_f1 = safe_ratio(2 * boundary_precision * boundary_recall, boundary_precision + boundary_recall)

    predicted_components, largest_fraction = filtered_component_count(prediction, minimum_component_pixels)
    truth_components, _ = filtered_component_count(truth, minimum_component_pixels)
    return {
        "cldice": cldice,
        "skeleton_precision": skeleton_precision,
        "skeleton_recall": skeleton_recall,
        "branchpoint_precision": branch_precision,
        "branchpoint_recall": branch_recall,
        "branchpoint_f1": branch_f1,
        "branchpoint_predicted_count": int(len(predicted_nodes)),
        "branchpoint_truth_count": int(len(truth_nodes)),
        "thin_centerline_recall": thin_recall,
        "medium_centerline_recall": medium_recall,
        "thick_centerline_recall": thick_recall,
        "boundary_f1_euclidean_2px": boundary_f1,
        "component_count_ge10": predicted_components,
        "component_excess_vs_truth_ge10": predicted_components - truth_components,
        "largest_component_fraction": largest_fraction,
    }


def index_files(directory: Path) -> dict[str, Path]:
    return {path.stem.lower(): path for path in directory.iterdir() if path.is_file()}


def structural_analysis(args: argparse.Namespace, metadata: dict[str, dict[str, str]]) -> list[dict]:
    ground_truth_files = index_files(args.fives_root / "test" / "Ground truth")
    original_files = index_files(args.fives_root / "test" / "Original")
    if set(ground_truth_files) != set(metadata) or set(original_files) != set(metadata):
        raise ValueError("FIVES official-test Original/Ground truth sets do not match metadata")
    require_workers = max(1, int(args.structure_workers))

    def analyze_sample(sample: str) -> list[dict]:
        sample_rows: list[dict] = []
        truth = resize_mask(ground_truth_files[sample], args.image_size)
        original = resize_rgb(original_files[sample], args.image_size)
        grayscale = np.asarray(original.convert("L"), dtype=np.float32) / 255.0
        fov = grayscale > (5.0 / 255.0)
        truth &= fov
        truth_skeleton = skeletonize_zhang_suen(truth)
        truth_radius = ndi.distance_transform_edt(truth)
        for model in MODELS:
            for seed in SEEDS:
                path = args.result_root / model / f"seed_{seed}" / "predictions" / f"{sample}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
                prediction = resize_mask(path, args.image_size)
                sample_rows.append(
                    {
                        "sample": sample,
                        "model": model,
                        "seed": seed,
                        "disease": metadata[sample]["Disease"],
                        "IC": int(metadata[sample]["IC"]),
                        "Blur": int(metadata[sample]["Blur"]),
                        "LC": int(metadata[sample]["LC"]),
                        **structural_metrics(
                            prediction,
                            truth,
                            fov,
                            truth_skeleton,
                            truth_radius,
                            args.boundary_tolerance,
                            args.junction_tolerance,
                            args.minimum_component_pixels,
                        ),
                    }
                )
        return sample_rows

    rows: list[dict] = []
    samples = sorted(metadata)
    with ThreadPoolExecutor(max_workers=require_workers) as executor:
        for position, sample_rows in enumerate(executor.map(analyze_sample, samples), start=1):
            rows.extend(sample_rows)
            if position % 10 == 0:
                print(f"Audited structural analysis: {position}/200 images", flush=True)
    return rows


def chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def error_panel(prediction: np.ndarray, truth: np.ndarray) -> Image.Image:
    canvas = np.zeros((*truth.shape, 3), dtype=np.uint8)
    canvas[prediction & truth] = (0, 190, 0)
    canvas[prediction & ~truth] = (230, 45, 45)
    canvas[~prediction & truth] = (45, 105, 235)
    return Image.fromarray(canvas)


def seed_metric_map(rows: list[dict], metric: str, seed: int) -> dict[tuple[str, str], float]:
    return {
        (row["model"], row["sample"]): float(row[metric])
        for row in rows
        if int(row["seed"]) == seed
    }


def select_audited_cases(rows: list[dict], seed: int = 42) -> list[tuple[str, str]]:
    values = seed_metric_map(rows, "dice", seed)
    samples = sorted({row["sample"] for row in rows})
    deltas = np.asarray(
        [values[("carer_unet", sample)] - values[("gray_rgb_control_30_unet", sample)] for sample in samples]
    )
    median = float(np.median(deltas))
    return [
        ("按固定seed=42选取：改善最大", samples[int(np.argmax(deltas))]),
        ("按固定seed=42选取：中位差值", samples[int(np.argmin(np.abs(deltas - median)))]),
        ("按固定seed=42选取：下降最大（失败例）", samples[int(np.argmin(deltas))]),
    ]


def load_prediction(result_root: Path, model: str, sample: str, size: int, seed: int = 42) -> np.ndarray:
    return resize_mask(result_root / model / f"seed_{seed}" / "predictions" / f"{sample}.png", size)


def choose_zoom_box(control: np.ndarray, carer: np.ndarray, truth: np.ndarray, box: int = 112) -> tuple[int, int, int, int]:
    informative = ((control != truth) | (carer != truth)) & (control != carer)
    if not informative.any():
        informative = control != truth
    density = ndi.uniform_filter(informative.astype(np.float32), size=box, mode="constant")
    row, column = np.unravel_index(int(np.argmax(density)), density.shape)
    half = box // 2
    top = max(0, min(truth.shape[0] - box, row - half))
    left = max(0, min(truth.shape[1] - box, column - half))
    return left, top, left + box, top + box


def create_case_figure(
    args: argparse.Namespace,
    metadata: dict[str, dict[str, str]],
    metric_rows_all: list[dict],
) -> Path:
    originals = index_files(args.fives_root / "test" / "Original")
    truths = index_files(args.fives_root / "test" / "Ground truth")
    values = seed_metric_map(metric_rows_all, "dice", 42)
    cases = select_audited_cases(metric_rows_all, 42)
    labels = (
        "原图（红框为放大区）",
        "金标准",
        "对照误差",
        "颜色模块误差",
        "CARER误差",
        "对照局部",
        "颜色模块局部",
        "CARER局部",
    )
    panel = 252
    top_header = 62
    row_header = 38
    title_font = chinese_font(19)
    small_font = chinese_font(16)
    canvas = Image.new("RGB", (len(labels) * panel, top_header + len(cases) * (panel + row_header)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * panel + 6, 10), label, fill="black", font=title_font)
    draw.text((6, 37), "误差图：绿=TP，红=FP，蓝=FN；显示结果与数值均固定为seed=42", fill=(60, 60, 60), font=small_font)
    for row_index, (selection_label, sample) in enumerate(cases):
        row_top = top_header + row_index * (panel + row_header)
        truth = resize_mask(truths[sample], args.image_size)
        control = load_prediction(args.result_root, "gray_rgb_control_30_unet", sample, args.image_size)
        color = load_prediction(args.result_root, "color_detail_only_unet", sample, args.image_size)
        carer = load_prediction(args.result_root, "carer_unet", sample, args.image_size)
        box = choose_zoom_box(control, carer, truth)
        original = resize_rgb(originals[sample], args.image_size)
        original_draw = ImageDraw.Draw(original)
        original_draw.rectangle(box, outline=(255, 35, 35), width=4)
        control_error = error_panel(control, truth)
        color_error = error_panel(color, truth)
        carer_error = error_panel(carer, truth)
        panels = (
            original,
            Image.fromarray(np.repeat((truth.astype(np.uint8) * 255)[..., None], 3, axis=2)),
            control_error,
            color_error,
            carer_error,
            control_error.crop(box),
            color_error.crop(box),
            carer_error.crop(box),
        )
        delta = values[("carer_unet", sample)] - values[("gray_rgb_control_30_unet", sample)]
        description = (
            f"{selection_label}；{sample}；{DISEASE_NAMES[metadata[sample]['Disease']]}；"
            f"CARER-对照 Dice={delta:+.4f}"
        )
        draw.text((7, row_top + 7), description, fill="black", font=small_font)
        for column, image in enumerate(panels):
            image = image.resize((panel, panel), Image.Resampling.NEAREST if column else Image.Resampling.BILINEAR)
            canvas.paste(image, (column * panel, row_top + row_header))
    path = args.output_dir / "图_FIVES代表性病例_中文.png"
    canvas.save(path)
    return path


def create_subgroup_figure(rows: list[dict], output_dir: Path) -> Path:
    selected = [
        row
        for row in rows
        if row["comparison"] == "CARER-计算匹配对照"
        and row["metric"] == "dice"
        and (
            row["group_type"] == "病种"
            or (row["group_type"] in {"照明/颜色", "模糊", "低对比"} and row["group"] == "质量较差")
        )
    ]
    labels = []
    for row in selected:
        label = row["group"] if row["group_type"] == "病种" else f"{row['group_type']}差"
        labels.append(f"{label} (n={row['n_images']})")
    y = np.arange(len(selected))
    values = np.asarray([float(row["mean_delta"]) for row in selected])
    lows = np.asarray([float(row["ci_low"]) for row in selected])
    highs = np.asarray([float(row["ci_high"]) for row in selected])
    colors = [
        "#4472C4" if low > 0 else "#C00000" if high < 0 else "#8A8A8A"
        for low, high in zip(lows, highs)
    ]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10.5, 6.1), dpi=180)
    for position, value, low, high, color in zip(y, values, lows, highs, colors):
        ax.errorbar(value, position, xerr=[[value - low], [high - value]], fmt="o", color=color, capsize=4, markersize=7)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("CARER 相对计算匹配对照的 Dice 差值")
    ax.set_title("FIVES 病种与质量分层（探索性；seed×图像双向Bootstrap 95% CI）")
    ax.grid(axis="x", alpha=0.25)
    ax.text(
        0.01,
        -0.16,
        "灰色表示区间跨0。质量分层受病种与质量属性重叠影响；低对比差组尤其不能作独立质量效应解释。",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout()
    path = output_dir / "图_FIVES病种与质量分层_中文.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def find_row(rows: list[dict], comparison: str, metric: str, group_type: str = "总体", group: str = "全部") -> dict:
    return next(
        row
        for row in rows
        if row["comparison"] == comparison
        and row["metric"] == metric
        and row["group_type"] == group_type
        and row["group"] == group
    )


def format_effect(row: dict) -> str:
    return f"{row['mean_delta']:+.6f} [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]"


def report_markdown(
    path: Path,
    subgroup_rows: list[dict],
    structural_rows: list[dict],
    interactions: list[dict],
    case_path: Path,
    subgroup_path: Path,
) -> None:
    primary_dice = find_row(subgroup_rows, "CARER-计算匹配对照", "dice")
    primary_cldice = find_row(structural_rows, "CARER-计算匹配对照", "cldice")
    lines = [
        "# CARER-Net JIIM 第一阶段审计后分析报告",
        "",
        "## 统计口径",
        "",
        "- 预先指定的两个主要终点为 FIVES 官方测试集总体 Dice 与 clDice，比较 CARER 和计算匹配对照。为使该两终点的家族错误率不超过 0.05，分别报告 97.5% 双侧区间。",
        "- 每次 Bootstrap 同时重采样 3 个训练随机种子和测试图像，保留 seed×图像的交叉变异；疾病、质量和其余结构指标均为探索性 pointwise 95% 区间。",
        "- 本地 FIVES 发布物没有 patient_id。FIVES 的图像可能来自同一受试者双眼，因此以下推断只能视为图像级、对当前三次训练条件成立的描述，不能替代患者聚类分析。",
        "- 不再把均值差 Bootstrap CI 与 Wilcoxon 秩检验混用；本报告不以探索性 p 值作确认性结论。",
        "",
        "## 主要终点",
        "",
        "| 终点 | CARER-对照均值差 | 97.5% seed×图像 Bootstrap CI | 三个种子差值 | 结论 |",
        "|---|---:|---:|---:|---|",
        f"| Dice | {primary_dice['mean_delta']:+.6f} | [{primary_dice['ci_low']:+.6f}, {primary_dice['ci_high']:+.6f}] | {primary_dice['seed_42_delta']:+.6f}; {primary_dice['seed_123_delta']:+.6f}; {primary_dice['seed_3407_delta']:+.6f} | {primary_dice['evidence']} |",
        f"| clDice | {primary_cldice['mean_delta']:+.6f} | [{primary_cldice['ci_low']:+.6f}, {primary_cldice['ci_high']:+.6f}] | {primary_cldice['seed_42_delta']:+.6f}; {primary_cldice['seed_123_delta']:+.6f}; {primary_cldice['seed_3407_delta']:+.6f} | {primary_cldice['evidence']} |",
        "",
        "## 疾病与图像质量分层（探索性）",
        "",
        "| 分层 | n | Dice差值 [95% CI] | seed 42/123/3407 |",
        "|---|---:|---:|---:|",
    ]
    for group_type, group in (
        ("病种", "AMD"), ("病种", "DR"), ("病种", "青光眼"), ("病种", "正常"),
        ("照明/颜色", "质量较差"), ("模糊", "质量较差"), ("低对比", "质量较差"),
    ):
        row = find_row(subgroup_rows, "CARER-计算匹配对照", "dice", group_type, group)
        label = group if group_type == "病种" else f"{group_type}-{group}"
        lines.append(
            f"| {label} | {row['n_images']} | {format_effect(row)} | "
            f"{row['seed_42_delta']:+.5f}/{row['seed_123_delta']:+.5f}/{row['seed_3407_delta']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "质量标签高度重叠且与病种构成不均衡。特别是低对比差组样本很少，不能把其分层结果解释为独立的低对比因果效应。下表仅给出病种指示变量调整后的 method×quality 交互探索。",
            "",
            "| 质量属性 | 差/好样本 | 病种调整后（差-好）增益差 [95% CI] | 说明 |",
            "|---|---:|---:|---|",
        ]
    )
    for row in interactions:
        lines.append(
            f"| {row['quality_label']} | {row['n_poor']}/{row['n_good']} | "
            f"{row['adjusted_interaction']:+.6f} [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | {row['poor_disease_counts']} |"
        )
    lines.extend(
        [
            "",
            "## 结构与误差机制（探索性）",
            "",
            "| 指标 | CARER-对照差值 [95% CI] | 三个种子差值 |",
            "|---|---:|---:|",
        ]
    )
    for metric in (
        "skeleton_precision", "skeleton_recall", "branchpoint_precision", "branchpoint_recall",
        "branchpoint_f1", "thin_centerline_recall", "medium_centerline_recall",
        "thick_centerline_recall", "boundary_f1_euclidean_2px", "component_excess_vs_truth_ge10",
        "largest_component_fraction",
    ):
        row = find_row(structural_rows, "CARER-计算匹配对照", metric)
        lines.append(
            f"| {metric} | {format_effect(row)} | "
            f"{row['seed_42_delta']:+.5f}/{row['seed_123_delta']:+.5f}/{row['seed_3407_delta']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "细/中/粗口径现在依据单像素 GT 骨架位置处的局部欧氏半径分层，而不是原实现中的边缘壳层。分叉点由单像素骨架的 junction 像素聚类成节点，再在 2 px 内一对一匹配。边界 F1 使用欧氏距离容差。连通分量仅统计不少于10像素的分量，并必须结合召回和最大分量比例解释；分量更少本身不等于更好。",
            "",
            "## 精修器相对颜色模块的独立贡献",
            "",
            "| 指标 | CARER-颜色模块差值 [95% CI] | 三个种子差值 |",
            "|---|---:|---:|",
        ]
    )
    for metric, source in (
        ("dice", subgroup_rows), ("sensitivity", subgroup_rows), ("specificity", subgroup_rows),
        ("cldice", structural_rows), ("skeleton_recall", structural_rows),
        ("thin_centerline_recall", structural_rows), ("boundary_f1_euclidean_2px", structural_rows),
    ):
        row = find_row(source, "CARER-颜色模块", metric)
        lines.append(
            f"| {metric} | {format_effect(row)} | "
            f"{row['seed_42_delta']:+.5f}/{row['seed_123_delta']:+.5f}/{row['seed_3407_delta']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "上述精修器分析不预设一定改善；是否保留或升级精修器，应结合三个种子方向和独立验证门控决定，不能只依据单个显著性标签。",
            "",
            "## 图件",
            "",
            f"- {subgroup_path.name}：探索性 forest plot，灰色表示区间跨0。",
            f"- {case_path.name}：固定 seed=42；案例按明确的最大/中位/最差规则选取，属于结果驱动的成功/失败示例，不称为随机代表病例。图中数值和掩膜来自同一 seed。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(args: argparse.Namespace, metadata_path: Path) -> dict:
    metric_files = [
        args.result_root / model / f"seed_{seed}" / "per_image_metrics.csv"
        for model in MODELS for seed in SEEDS
    ]
    prediction_files = [
        args.result_root / model / f"seed_{seed}" / "predictions" / f"{sample}.png"
        for model in MODELS for seed in SEEDS for sample in sorted(load_metadata(metadata_path))
    ]
    gt_root = args.fives_root / "test" / "Ground truth"
    gt_files = [path for path in gt_root.iterdir() if path.is_file()]
    return {
        "analysis": "CARER JIIM stage 1 audited revision",
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "statistical_design": {
            "crossed_clusters": ["training_seed", "test_image"],
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "primary_family": ["overall Dice: CARER vs compute-matched control", "overall clDice: CARER vs compute-matched control"],
            "primary_interval": "two-sided 97.5% percentile two-way bootstrap (Bonferroni family of 2)",
            "exploratory_interval": "two-sided pointwise 95% percentile two-way bootstrap",
            "p_values": "not reported; avoids mixing rank estimand with mean-difference CI",
            "patient_clustering": "unavailable because local metadata has no patient_id",
        },
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "image_size": args.image_size,
        "fov": {"source": "resized RGB converted to PIL grayscale", "threshold": 5.0 / 255.0},
        "structure": {
            "execution": f"deterministic ordered ThreadPoolExecutor with {max(1, int(args.structure_workers))} workers",
            "skeleton": "Zhang-Suen two-subiteration one-pixel thinning",
            "calibre": "GT skeleton local radius from Euclidean distance transform; thin<=1.5, medium(1.5,3], thick>3 px",
            "junction": f"8-neighbour degree>=3; connected junction pixels collapsed to one node; greedy one-to-one Euclidean match <= {args.junction_tolerance}px",
            "boundary": f"Euclidean distance tolerance <= {args.boundary_tolerance}px",
            "components": f"8-connected components with >= {args.minimum_component_pixels} pixels",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
        },
        "input_hashes": {
            "metadata_csv": sha256_file(metadata_path),
            "per_image_metrics_aggregate": aggregate_hash(metric_files, args.result_root),
            "binary_predictions_aggregate": aggregate_hash(prediction_files, args.result_root),
            "ground_truth_aggregate": aggregate_hash(gt_files, args.fives_root),
        },
        "quality_warning": "Quality attributes overlap and are disease-confounded; subgroup and adjusted interactions are exploratory.",
        "uncertainty_calibration": "not available in stage 1 because probability maps were not transferred; stage 2 exports them",
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1000:
        raise ValueError("Use at least 1000 bootstrap replicates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata_csv)
    rng = np.random.default_rng(args.bootstrap_seed)

    metrics = metric_rows(args.result_root, metadata)
    subgroup_rows = paired_analysis(
        metrics,
        metadata,
        ("dice", "sensitivity", "specificity", "iou"),
        args.bootstrap,
        rng,
    )
    write_csv(args.output_dir / "FIVES_病种质量分层统计.csv", subgroup_rows)

    interactions = disease_adjusted_quality_interactions(
        metrics, metadata, args.bootstrap, rng
    )
    write_csv(args.output_dir / "FIVES_质量交互病种调整.csv", interactions)

    structural = structural_analysis(args, metadata)
    write_csv(args.output_dir / "FIVES_逐图结构指标.csv", structural)
    structural_rows = paired_analysis(
        structural,
        metadata,
        (
            "cldice", "skeleton_precision", "skeleton_recall",
            "branchpoint_precision", "branchpoint_recall", "branchpoint_f1",
            "thin_centerline_recall", "medium_centerline_recall", "thick_centerline_recall",
            "boundary_f1_euclidean_2px", "component_count_ge10",
            "component_excess_vs_truth_ge10", "largest_component_fraction",
        ),
        args.bootstrap,
        rng,
    )
    write_csv(args.output_dir / "FIVES_结构指标配对统计.csv", structural_rows)

    subgroup_figure = create_subgroup_figure(subgroup_rows, args.output_dir)
    case_figure = create_case_figure(args, metadata, metrics)
    report_markdown(
        args.output_dir / "CARER_JIIM_第一阶段分析报告.md",
        subgroup_rows,
        structural_rows,
        interactions,
        case_figure,
        subgroup_figure,
    )
    manifest = build_manifest(args, args.metadata_csv)
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Audited stage-1 analysis complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
