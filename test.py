import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import DATASET_CHOICES, RetinalDataset, discover_auxiliary, discover_pairs
from evaluation import METRIC_NAMES, calculate_metrics, summarize_metrics
from models.model_factory import MODEL_CHOICES, build_model, checkpoint_tag, count_parameters


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on a held-out retinal test set.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="CHASE")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="improved_attention_unet")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--weight-dir", type=Path, default=PROJECT_ROOT / "weight_files")
    parser.add_argument("--result-root", type=Path, default=PROJECT_ROOT / "result")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--input-mode", choices=("grayscale", "green", "gray_rgb"), default="grayscale"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help="Choose the binary threshold on this run's validation split before testing.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--fov", choices=("derived", "files", "full"), default="derived")
    parser.add_argument("--fov-dir", type=Path, help="Official FOV masks, required with --fov files.")
    parser.add_argument("--fov-threshold", type=float, default=5 / 255)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=5,
        help="Unmeasured forward passes before CUDA timing.",
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def resolve_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint
    tag = checkpoint_tag(args.model)
    seeded = args.weight_dir / f"{args.dataset}_best_{tag}_seed{args.seed}.pth"
    legacy = args.weight_dir / f"{args.dataset}_best_{tag}.pth"
    if seeded.exists():
        return seeded
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"No checkpoint found. Checked {seeded} and {legacy}")


def load_checkpoint(model, path, device, allow_partial=False):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if allow_partial:
        model_state = model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(f"Partial checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        model.load_state_dict(state_dict, strict=True)
    return checkpoint


def write_json(path, value):
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validation_pairs_from_split(args):
    split_path = (
        args.run_root / args.dataset / args.model / f"seed_{args.seed}" / "split.json"
    )
    if not split_path.is_file():
        raise FileNotFoundError(f"Validation split not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as file:
        validation_keys = set(json.load(file)["validation"])
    train_pairs = discover_pairs(args.data_root, args.dataset, "train")
    selected = [pair for pair in train_pairs if pair[2] in validation_keys]
    selected_keys = {pair[2] for pair in selected}
    if selected_keys != validation_keys:
        missing = sorted(validation_keys - selected_keys)
        raise ValueError(f"Validation samples missing from training pool: {missing}")
    return selected, split_path


def collect_validation_predictions(model, pairs, args, device):
    dataset = RetinalDataset(pairs, args.image_size, args.input_mode, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    records = []
    with torch.inference_mode():
        for images, masks, keys in loader:
            probability = torch.sigmoid(model(images.to(device, non_blocking=True)))
            probability = probability.cpu().numpy()[0, 0]
            ground_truth = masks.numpy()[0, 0] > 0.5
            fov_mask = None
            if args.fov == "derived":
                fov_mask = images.numpy()[0, 0] > args.fov_threshold
            records.append((keys[0], ground_truth, probability, fov_mask))
    return records


def mean_validation_dice(records, threshold, eps=1e-8):
    values = []
    for _, ground_truth, probability, fov_mask in records:
        if fov_mask is None:
            fov_mask = np.ones_like(ground_truth, dtype=bool)
        target = ground_truth[fov_mask]
        prediction = probability[fov_mask] > threshold
        intersection = np.logical_and(target, prediction).sum()
        denominator = target.sum() + prediction.sum()
        values.append((2.0 * intersection + eps) / (denominator + eps))
    return float(np.mean(values))


def calibrate_threshold(model, args, device):
    if not 0.0 <= args.threshold_min < args.threshold_max <= 1.0:
        raise ValueError("Threshold calibration range must satisfy 0 <= min < max <= 1")
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be positive")
    validation_pairs, split_path = validation_pairs_from_split(args)
    records = collect_validation_predictions(model, validation_pairs, args, device)
    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2.0,
        args.threshold_step,
    )
    thresholds = np.clip(thresholds, args.threshold_min, args.threshold_max)
    scores = np.asarray(
        [mean_validation_dice(records, float(threshold)) for threshold in thresholds]
    )
    best_score = float(scores.max())
    tied = thresholds[np.isclose(scores, best_score, rtol=0.0, atol=1e-12)]
    selected = float(tied[np.argmin(np.abs(tied - 0.5))])
    return selected, {
        "method": "validation_mean_dice_grid_search",
        "selected_threshold": selected,
        "best_validation_dice": best_score,
        "validation_samples": [record[0] for record in records],
        "split_file": str(split_path.resolve()),
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_step": args.threshold_step,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint(args)
    pairs = discover_pairs(args.data_root, args.dataset, "test")
    fov_files = None
    if args.fov == "files":
        if args.fov_dir is None:
            raise ValueError("--fov-dir is required when --fov files is selected")
        fov_files = discover_auxiliary(args.fov_dir, args.dataset)
        missing = sorted({pair[2] for pair in pairs} - set(fov_files))
        if missing:
            raise ValueError(f"Missing official FOV masks for samples: {missing}")
    dataset = RetinalDataset(pairs, args.image_size, args.input_mode, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    gray_rgb_models = {
        "gray_rgb_control_unet",
        "gray_rgb_control_30_unet",
        "carer_unet",
        "color_detail_only_unet",
        "signed_refiner_only_unet",
    }
    if args.model in gray_rgb_models and args.input_mode != "gray_rgb":
        raise ValueError(f"Model {args.model} requires --input-mode gray_rgb")
    if args.input_mode == "gray_rgb" and args.model not in gray_rgb_models:
        raise ValueError("--input-mode gray_rgb is reserved for the CARER paired models")
    input_channels = 4 if args.input_mode == "gray_rgb" else 1
    model = build_model(args.model, in_channels=input_channels, out_channels=1).to(device)
    load_checkpoint(model, checkpoint_path, device, args.allow_partial_checkpoint)
    model.eval()

    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if len(dataset) and args.warmup_iterations:
        warmup_image = dataset[0][0].unsqueeze(0).to(device)
        with torch.inference_mode():
            for _ in range(args.warmup_iterations):
                model(warmup_image)
        if device.type == "cuda":
            torch.cuda.synchronize()

    selected_threshold = args.threshold
    threshold_selection = None
    if args.calibrate_threshold:
        if args.fov == "files":
            raise ValueError("Validation threshold calibration does not support --fov files")
        selected_threshold, threshold_selection = calibrate_threshold(model, args, device)
        print(
            f"Validation-selected threshold: {selected_threshold:.4f} "
            f"(validation Dice={threshold_selection['best_validation_dice']:.4f})"
        )

    output_dir = args.result_root / args.dataset / args.model / f"seed_{args.seed}"
    prediction_dir = output_dir / "predictions"
    probability_dir = output_dir / "probabilities"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if args.save_probabilities:
        probability_dir.mkdir(parents=True, exist_ok=True)
    if threshold_selection is not None:
        write_json(output_dir / "threshold_selection.json", threshold_selection)

    rows = []
    inference_times = []
    with torch.inference_mode():
        for images, masks, keys in tqdm(loader, desc="Testing", ncols=100):
            images_device = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images_device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - start)

            probability = torch.sigmoid(logits).cpu().numpy()[0, 0]
            ground_truth = masks.numpy()[0, 0] > 0.5
            fov_mask = None
            if args.fov == "derived":
                fov_mask = images.numpy()[0, 0] > args.fov_threshold
            elif args.fov == "files":
                fov_image = Image.open(fov_files[keys[0]]).convert("L")
                fov_image = fov_image.resize(
                    (args.image_size, args.image_size), Image.Resampling.NEAREST
                )
                fov_mask = np.asarray(fov_image, dtype=np.uint8) > 127
            metrics = calculate_metrics(
                ground_truth, probability, threshold=selected_threshold, fov_mask=fov_mask
            )
            rows.append(
                {
                    "sample": keys[0],
                    **metrics,
                    "inference_seconds": inference_times[-1],
                }
            )

            prediction = (probability > selected_threshold).astype(np.uint8) * 255
            Image.fromarray(prediction).save(prediction_dir / f"{keys[0]}.png")
            if args.save_probabilities:
                probability_16bit = np.round(probability * 65535).astype(np.uint16)
                Image.fromarray(probability_16bit).save(probability_dir / f"{keys[0]}.png")

    summary = summarize_metrics(rows)
    summary["inference_seconds"] = {
        "mean": float(np.mean(inference_times)),
        "std": float(np.std(inference_times, ddof=1)) if len(inference_times) > 1 else 0.0,
    }
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "test_samples": len(dataset),
        "image_size": args.image_size,
        "input_mode": args.input_mode,
        "threshold": selected_threshold,
        "threshold_source": "validation" if args.calibrate_threshold else "fixed",
        "threshold_selection": threshold_selection,
        "fov": args.fov,
        "parameter_count": count_parameters(model),
        "timing_warmup_iterations": args.warmup_iterations,
        "metrics": summary,
    }

    with (output_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "metrics.json", metadata)

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as file:
        file.write(f"{args.dataset} test metrics\n")
        file.write("=" * 52 + "\n")
        for name in METRIC_NAMES:
            values = summary[name]
            file.write(f"{name:16s}: {values['mean']:.4f} ± {values['std']:.4f}\n")
        timing = summary["inference_seconds"]
        file.write(f"inference_seconds: {timing['mean']:.4f} ± {timing['std']:.4f}\n")

    print(f"Evaluation complete: {output_dir}")
    for name in METRIC_NAMES:
        values = summary[name]
        print(f"{name:16s}: {values['mean']:.4f} ± {values['std']:.4f}")


if __name__ == "__main__":
    main()
