import argparse
import gc
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from models.model_factory import build_model, count_parameters


DATASETS = ("DRIVE", "STARE", "CHASE")
SEEDS = (42, 123, 3407)
MODEL_SPECS = (
    {
        "model": "res_attention_unet_gn",
        "checkpoint_model": "res_attention_unet_gn",
        "input_mode": "grayscale",
        "filename": "{dataset}_best_ResAttentionUNetGN_seed{seed}.pth",
        "source": "gn",
    },
    {
        "model": "gray_rgb_control_30_unet",
        "checkpoint_model": "gray_rgb_control_30_unet",
        "input_mode": "gray_rgb",
        "filename": "{dataset}_best_gray_rgb_control_30_unet_seed{seed}.pth",
        "source": "carer",
    },
    {
        "model": "color_detail_only_unet",
        "checkpoint_model": "color_detail_only_unet",
        "input_mode": "gray_rgb",
        "filename": "{dataset}_best_color_detail_only_unet_seed{seed}.pth",
        "source": "carer",
    },
    {
        "model": "carer_unet",
        "checkpoint_model": "carer_unet",
        "input_mode": "gray_rgb",
        "filename": "{dataset}_best_carer_unet_seed{seed}.pth",
        "source": "carer",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strictly verify all three-dataset CARER checkpoints."
    )
    parser.add_argument("--gn-weight-dir", type=Path, required=True)
    parser.add_argument("--carer-weight-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def checkpoint_seed(checkpoint):
    config = checkpoint.get("config") or {}
    for source in (checkpoint, config):
        for key in ("train_seed", "seed"):
            if source.get(key) is not None:
                return int(source[key])
    return None


def validate(path, dataset, seed, spec):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no state_dict: {path}")
    if checkpoint.get("dataset") not in (None, dataset):
        raise ValueError(f"Dataset mismatch in {path}: {checkpoint.get('dataset')}")
    if checkpoint.get("model_name") not in (None, spec["checkpoint_model"]):
        raise ValueError(f"Model mismatch in {path}: {checkpoint.get('model_name')}")
    recorded_seed = checkpoint_seed(checkpoint)
    if recorded_seed is not None and recorded_seed != seed:
        raise ValueError(f"Seed mismatch in {path}: expected {seed}, found {recorded_seed}")
    config = checkpoint.get("config") or {}
    if int(config.get("image_size", 512)) != 512:
        raise ValueError(f"Image size mismatch in {path}")
    input_mode = checkpoint.get("input_mode", config.get("input_mode"))
    if input_mode is not None and input_mode != spec["input_mode"]:
        raise ValueError(
            f"Input mode mismatch in {path}: expected {spec['input_mode']}, found {input_mode}"
        )
    channels = 4 if spec["input_mode"] == "gray_rgb" else 1
    model = build_model(spec["model"], in_channels=channels, out_channels=1)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    record = {
        "dataset": dataset,
        "model": spec["model"],
        "seed": seed,
        "input_mode": spec["input_mode"],
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path),
        "parameter_count": count_parameters(model),
        "strict_state_dict_load": True,
    }
    del model, checkpoint
    gc.collect()
    return record


def main():
    args = parse_args()
    records = []
    planned = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for spec in MODEL_SPECS:
                root = args.gn_weight_dir if spec["source"] == "gn" else args.carer_weight_dir
                path = root / spec["filename"].format(dataset=dataset, seed=seed)
                planned.append((path, dataset, seed, spec))
    missing = [str(path) for path, _, _, _ in planned if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    for index, (path, dataset, seed, spec) in enumerate(planned, 1):
        print(f"[{index}/{len(planned)}] {dataset} {spec['model']} seed {seed}")
        records.append(validate(path, dataset, seed, spec))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "models": [spec["model"] for spec in MODEL_SPECS],
        "checkpoints": records,
        "strict_state_dict_load": True,
        "checkpoint_count": len(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"Verified {len(records)} checkpoints. Manifest: {args.output}")


if __name__ == "__main__":
    main()
