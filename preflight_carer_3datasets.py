import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from data_utils import discover_pairs
from models.model_factory import build_model, count_parameters
from train_carer_paired import load_explicit_split


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS = ("DRIVE", "STARE", "CHASE")
SEEDS = (42, 123, 3407)
EXPECTED_COUNTS = {
    "DRIVE": {"train": 20, "test": 20},
    "STARE": {"train": 10, "test": 10},
    "CHASE": {"train": 20, "test": 8},
}
MODELS = (
    ("res_attention_unet_gn", "grayscale"),
    ("gray_rgb_control_30_unet", "gray_rgb"),
    ("color_detail_only_unet", "gray_rgb"),
    ("carer_unet", "gray_rgb"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preflight the frozen CARER three-dataset experiment."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--decision-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = (args.data_root or project_root / "data").resolve()
    split_root = (
        args.split_root or project_root / "reference_splits_carer_3datasets"
    ).resolve()
    decision_file = (
        args.decision_file
        or project_root
        / "frozen_protocol_carer_3datasets"
        / "FIVES_carer_multiseed_decision.json"
    ).resolve()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is False")
    if not decision_file.is_file():
        raise FileNotFoundError(f"Frozen FIVES decision not found: {decision_file}")
    decision = read_json(decision_file)
    if decision.get("verdict") != "SELECT_FULL_CARER":
        raise RuntimeError(
            f"Expected verdict SELECT_FULL_CARER, found {decision.get('verdict')}"
        )
    if decision.get("selected_model") != "carer_unet":
        raise RuntimeError(
            f"Expected selected_model=carer_unet, found {decision.get('selected_model')}"
        )

    split_records = []
    dataset_records = []
    for dataset in DATASETS:
        train_pairs = discover_pairs(data_root, dataset, "train")
        test_pairs = discover_pairs(data_root, dataset, "test")
        expected = EXPECTED_COUNTS[dataset]
        if len(train_pairs) != expected["train"] or len(test_pairs) != expected["test"]:
            raise RuntimeError(
                f"{dataset} expected {expected['train']} train / {expected['test']} test "
                f"pairs, found {len(train_pairs)} train / {len(test_pairs)} test"
            )
        dataset_records.append(
            {"dataset": dataset, "train_pairs": len(train_pairs), "test_pairs": len(test_pairs)}
        )
        for seed in SEEDS:
            split_file = split_root / dataset / f"seed_{seed}.json"
            if not split_file.is_file():
                raise FileNotFoundError(f"Reference split not found: {split_file}")
            selected_train, selected_validation, _ = load_explicit_split(
                train_pairs, split_file, dataset
            )
            split_records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "split_file": str(split_file),
                    "train_samples": len(selected_train),
                    "validation_samples": len(selected_validation),
                }
            )

    model_records = []
    for model_name, input_mode in MODELS:
        input_channels = 4 if input_mode == "gray_rgb" else 1
        model = build_model(model_name, in_channels=input_channels, out_channels=1)
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, input_channels, 64, 64))
        if tuple(output.shape) != (1, 1, 64, 64):
            raise RuntimeError(f"Unexpected output shape for {model_name}: {tuple(output.shape)}")
        model_records.append(
            {
                "model": model_name,
                "input_mode": input_mode,
                "parameter_count": count_parameters(model),
                "forward_shape": list(output.shape),
            }
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_root": str(data_root),
        "frozen_decision": str(decision_file),
        "frozen_verdict": decision["verdict"],
        "selected_model": decision["selected_model"],
        "datasets": dataset_records,
        "splits": split_records,
        "models": model_records,
        "protocol": (
            "Frozen CARER architecture and FIVES-selected hyperparameters; explicit historical "
            "per-seed splits; three seeds; fixed threshold 0.5; derived FOV."
        ),
        "preflight_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print("CARER three-dataset preflight passed.")
    for record in dataset_records:
        print(
            f"{record['dataset']}: train={record['train_pairs']}, "
            f"test={record['test_pairs']}"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
