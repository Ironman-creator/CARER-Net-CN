from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from data_utils import discover_pairs
from models.model_factory import build_model, count_parameters


DATASETS = ("DRIVE", "STARE", "CHASE", "FIVES")
MODELS = ("unet", "attention_unet", "unet_plus_plus_gn", "r2_unet_gn", "sa_unet_gn")
REFERENCE_MODELS = (
    "res_attention_unet_gn",
    "gray_rgb_control_30_unet",
    "color_detail_only_unet",
    "carer_unet",
)
SEEDS = (42, 123, 3407)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def validate_split(path: Path, train_keys: set[str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    training = set(payload.get("train", payload.get("training", [])))
    validation = set(payload.get("validation", payload.get("val", [])))
    if not training or not validation:
        raise ValueError(f"Split lacks train/validation entries: {path}")
    if training & validation:
        raise ValueError(f"Split overlap in {path}")
    if training | validation != train_keys:
        missing = sorted(train_keys - (training | validation))
        extra = sorted((training | validation) - train_keys)
        raise ValueError(f"Split mismatch in {path}: missing={missing}, extra={extra}")
    return {"train": len(training), "validation": len(validation), "path": str(path)}


def resolve_reference_checkpoint(root: Path, dataset: str, model: str, seed: int) -> Path:
    if dataset == "FIVES":
        if model == "res_attention_unet_gn":
            return root / "weight_files_fives_v2_screen" / f"FIVES_best_ResAttentionUNetGN_seed{seed}.pth"
        if seed == 42:
            if model in {"gray_rgb_control_30_unet", "color_detail_only_unet"}:
                return root / "weight_files_fives_carer_ablation" / f"FIVES_best_{model}_seed42.pth"
            return root / "weight_files_fives_carer_paired_screen" / "FIVES_best_carer_unet_seed42.pth"
        return root / "weight_files_fives_carer_multiseed_confirm" / f"FIVES_best_{model}_seed{seed}.pth"
    if model == "res_attention_unet_gn":
        return root / "weight_files_carer_3datasets_gn" / f"{dataset}_best_ResAttentionUNetGN_seed{seed}.pth"
    return root / "weight_files_carer_3datasets_confirm" / f"{dataset}_best_{model}_seed{seed}.pth"


def validate_checkpoint(path: Path, dataset: str, model_name: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Checkpoint lacks state_dict: {path}")
    if payload.get("dataset") not in (None, dataset):
        raise ValueError(f"Checkpoint dataset mismatch: {path}")
    if payload.get("model_name") not in (None, model_name):
        raise ValueError(f"Checkpoint model mismatch: {path}")
    input_mode = "gray_rgb" if model_name != "res_attention_unet_gn" else "grayscale"
    model = build_model(model_name, in_channels=4 if input_mode == "gray_rgb" else 1, out_channels=1)
    model.load_state_dict(payload["state_dict"], strict=True)
    parameter_count = count_parameters(model)
    del model, payload
    return {
        "path": str(path),
        "dataset": dataset,
        "model": model_name,
        "input_mode": input_mode,
        "parameter_count": parameter_count,
    }


def main():
    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")

    datasets = {}
    for dataset in DATASETS:
        training_pairs = discover_pairs(args.data_root, dataset, "train")
        test_pairs = discover_pairs(args.data_root, dataset, "test")
        keys = {key for _, _, key in training_pairs}
        splits = {}
        for seed in SEEDS:
            path = args.split_root / dataset / f"seed_{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            splits[str(seed)] = validate_split(path, keys)
        datasets[dataset] = {
            "train_pairs": len(training_pairs),
            "test_pairs": len(test_pairs),
            "splits": splits,
        }

    checkpoint_inventory = []
    for dataset in DATASETS:
        for model_name in REFERENCE_MODELS:
            for seed in SEEDS:
                path = resolve_reference_checkpoint(
                    args.project_root, dataset, model_name, seed
                )
                # Strict-load every FIVES checkpoint because zero-shot external
                # validation depends on all twelve.  For the three legacy
                # datasets, strict-load seed 42 and still require all files.
                if dataset == "FIVES" or seed == 42:
                    checkpoint_inventory.append(
                        validate_checkpoint(path, dataset, model_name)
                    )
                elif not path.is_file():
                    raise FileNotFoundError(path)
                else:
                    checkpoint_inventory.append(
                        {
                            "path": str(path),
                            "dataset": dataset,
                            "model": model_name,
                            "input_mode": "gray_rgb" if model_name != "res_attention_unet_gn" else "grayscale",
                            "strict_loaded": False,
                        }
                    )

    forwards = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in (*MODELS, *REFERENCE_MODELS):
        gray_rgb = name in {
            "gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet"
        }
        sample = torch.rand(1, 4 if gray_rgb else 1, 512, 512, device=device)
        model = build_model(
            name, in_channels=4 if gray_rgb else 1, out_channels=1
        ).to(device).eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output = model(sample)
        if tuple(output.shape) != (1, 1, 512, 512):
            raise RuntimeError(f"Unexpected {name} output: {tuple(output.shape)}")
        forwards[name] = {
            "parameters": count_parameters(model),
            "shape": list(output.shape),
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
            ),
        }
        del model, sample, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    report = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "datasets": datasets,
        "checkpoint_inventory": checkpoint_inventory,
        "model_forwards": forwards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("JIIM stage-2 preflight passed")


if __name__ == "__main__":
    main()
