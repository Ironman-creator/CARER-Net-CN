"""Uniform RTX efficiency benchmark for all stage-2 comparison models."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from models.model_factory import build_model, count_parameters


MODELS = (
    "unet",
    "attention_unet",
    "unet_plus_plus_gn",
    "r2_unet_gn",
    "sa_unet_gn",
    "res_attention_unet_gn",
    "gray_rgb_control_30_unet",
    "color_detail_only_unet",
    "carer_unet",
)
GRAY_RGB = {"gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the efficiency benchmark")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    rows = []
    try:
        from thop import profile
    except ImportError:
        profile = None
    for name in MODELS:
        channels = 4 if name in GRAY_RGB else 1
        model = build_model(name, in_channels=channels, out_channels=1).to(device).eval()
        sample = torch.rand(1, channels, args.image_size, args.image_size, device=device)
        macs = None
        if profile is not None:
            cpu_model = build_model(name, in_channels=channels, out_channels=1).eval()
            cpu_sample = torch.rand(1, channels, args.image_size, args.image_size)
            macs, _ = profile(cpu_model, inputs=(cpu_sample,), verbose=False)
            del cpu_model, cpu_sample
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(sample)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        with torch.inference_mode():
            for _ in range(args.iterations):
                start = time.perf_counter()
                model(sample)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        peak = int(torch.cuda.max_memory_allocated())
        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        rows.append(
            {
                "model": name,
                "input_channels": channels,
                "image_size": args.image_size,
                "batch_size": 1,
                "parameters": count_parameters(model),
                "macs": int(macs) if macs is not None else "",
                "approx_flops_2x_macs": int(2 * macs) if macs is not None else "",
                "peak_cuda_allocated_bytes": peak,
                "latency_seconds_mean": mean,
                "latency_seconds_std": std,
                "fps": 1.0 / mean,
                "warmup_iterations": args.warmup,
                "timed_iterations": args.iterations,
                "gpu": torch.cuda.get_device_name(0),
                "pytorch": torch.__version__,
            }
        )
        del model, sample
        torch.cuda.empty_cache()
    with (args.output_dir / "JIIM_efficiency_benchmark.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "protocol": "single-image 512x512 CUDA inference; 20 warm-up and 100 synchronized timed iterations by default",
        "macs": "THOP static MACs when thop is installed; FLOPs shown only as the explicit 2x-MAC convention",
        "weights": "random weights; architecture execution cost is independent of trained values",
        "rows": rows,
    }
    (args.output_dir / "JIIM_efficiency_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Efficiency benchmark written to {args.output_dir}")


if __name__ == "__main__":
    main()
