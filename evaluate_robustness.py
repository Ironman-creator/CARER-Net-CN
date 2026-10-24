"""Deterministic FIVES corruption evaluation for a frozen checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils import discover_pairs
from evaluation import METRIC_NAMES, calculate_metrics, summarize_metrics
from models.model_factory import MODEL_CHOICES, build_model, count_parameters
from test import file_sha256, load_checkpoint, resolve_device


CORRUPTIONS = (
    "clean",
    "dark_0.65",
    "bright_1.35",
    "contrast_0.65",
    "gamma_1.50",
    "gaussian_blur_1.5",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--corruption", choices=CORRUPTIONS, required=True)
    parser.add_argument("--input-mode", choices=("grayscale", "gray_rgb"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-iterations", type=int, default=5)
    return parser.parse_args()


def corrupt(image: Image.Image, name: str) -> Image.Image:
    if name == "clean":
        return image
    if name == "dark_0.65":
        return ImageEnhance.Brightness(image).enhance(0.65)
    if name == "bright_1.35":
        return ImageEnhance.Brightness(image).enhance(1.35)
    if name == "contrast_0.65":
        return ImageEnhance.Contrast(image).enhance(0.65)
    if name == "gamma_1.50":
        lookup = [round(255 * ((value / 255) ** 1.5)) for value in range(256)]
        return image.point(lookup * 3)
    if name == "gaussian_blur_1.5":
        return image.filter(ImageFilter.GaussianBlur(radius=1.5))
    raise ValueError(name)


class RobustnessDataset(Dataset):
    def __init__(self, pairs, image_size: int, input_mode: str, corruption: str):
        self.pairs = list(pairs)
        self.size = (image_size, image_size)
        self.input_mode = input_mode
        self.corruption = corruption

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path, key = self.pairs[index]
        clean = Image.open(image_path).convert("RGB").resize(self.size, Image.Resampling.BILINEAR)
        altered = corrupt(clean, self.corruption)
        mask = Image.open(mask_path).convert("L").resize(self.size, Image.Resampling.NEAREST)
        clean_gray = np.asarray(clean.convert("L"), dtype=np.float32) / 255.0
        rgb = np.asarray(altered, dtype=np.float32) / 255.0
        gray = np.asarray(altered.convert("L"), dtype=np.float32) / 255.0
        if self.input_mode == "gray_rgb":
            inputs = np.concatenate((gray[..., None], rgb), axis=2)
            image_tensor = torch.from_numpy(inputs.copy()).permute(2, 0, 1)
        else:
            image_tensor = torch.from_numpy(gray.copy()).unsqueeze(0)
        target = np.asarray(mask, dtype=np.uint8) > 127
        return (
            image_tensor,
            torch.from_numpy(target.copy()).unsqueeze(0),
            torch.from_numpy((clean_gray > (5.0 / 255.0)).copy()).unsqueeze(0),
            key,
        )


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    pairs = discover_pairs(args.data_root, "FIVES", "test")
    dataset = RobustnessDataset(pairs, args.image_size, args.input_mode, args.corruption)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    input_channels = 4 if args.input_mode == "gray_rgb" else 1
    model = build_model(args.model, in_channels=input_channels, out_channels=1).to(device)
    load_checkpoint(model, args.checkpoint, device, allow_partial=False)
    model.eval()

    warmup = dataset[0][0].unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(args.warmup_iterations):
            model(warmup)
    if device.type == "cuda":
        torch.cuda.synchronize()

    rows = []
    elapsed = 0.0
    with torch.inference_mode():
        for images, masks, fovs, keys in tqdm(loader, desc=args.corruption, ncols=100):
            images_device = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            probability = torch.sigmoid(model(images_device)).cpu().numpy()[:, 0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start
            elapsed += batch_elapsed
            for position, key in enumerate(keys):
                metrics = calculate_metrics(
                    masks.numpy()[position, 0] > 0.5,
                    probability[position],
                    threshold=0.5,
                    fov_mask=fovs.numpy()[position, 0].astype(bool),
                )
                rows.append(
                    {
                        "sample": key,
                        **metrics,
                        "inference_seconds": batch_elapsed / len(keys),
                    }
                )

    output = args.result_root / args.corruption / args.model / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_metrics(rows)
    summary["inference_seconds"] = {
        "mean": elapsed / len(rows),
        "std": float(np.std([row["inference_seconds"] for row in rows], ddof=1)),
    }
    payload = {
        "dataset": "FIVES",
        "model": args.model,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "corruption": args.corruption,
        "corruption_scope": "RGB image only; FOV is always derived from the clean image",
        "threshold": 0.5,
        "test_samples": len(rows),
        "batch_size": args.batch_size,
        "parameter_count": count_parameters(model),
        "metrics": summary,
    }
    write_json(output / "metrics.json", payload)
    print(f"Robustness evaluation complete: {output}")


if __name__ == "__main__":
    main()
