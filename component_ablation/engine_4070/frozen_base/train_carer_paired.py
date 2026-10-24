import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import RetinalDataset, discover_pairs, seed_worker
from models.Attention_UNet_V5 import CARERUNet, GrayRGBControlUNet
from models.model_factory import count_parameters


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ("gray_rgb_control_unet", "carer_unet")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired GN fine-tuning control versus CARER validation screen."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--run-root", type=Path, default=PROJECT_ROOT / "runs_fives_carer_paired_screen"
    )
    parser.add_argument(
        "--weight-dir",
        type=Path,
        default=PROJECT_ROOT / "weight_files_fives_carer_paired_screen",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--tail-learning-rate", type=float, default=1e-5)
    parser.add_argument("--base-loss-weight", type=float, default=0.25)
    parser.add_argument("--error-loss-weight", type=float, default=0.10)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_explicit_split(all_pairs, split_file, dataset_name="FIVES"):
    with Path(split_file).open("r", encoding="utf-8-sig") as file:
        split = json.load(file)
    train_keys = list(split.get("train", []))
    validation_keys = list(split.get("validation", []))
    if not train_keys or not validation_keys:
        raise ValueError("split.json must contain non-empty train and validation lists")
    if len(set(train_keys)) != len(train_keys) or len(set(validation_keys)) != len(
        validation_keys
    ):
        raise ValueError("split.json contains duplicate sample keys")
    overlap = sorted(set(train_keys) & set(validation_keys))
    if overlap:
        raise ValueError(f"Training/validation overlap in split.json: {overlap[:10]}")

    by_key = {key: (image, mask, key) for image, mask, key in all_pairs}
    requested = set(train_keys) | set(validation_keys)
    missing = sorted(requested - set(by_key))
    extra = sorted(set(by_key) - requested)
    if missing or extra:
        raise ValueError(
            f"Explicit split does not exactly cover prepared {dataset_name} train data; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    train_pairs = [by_key[key] for key in train_keys]
    validation_pairs = [by_key[key] for key in validation_keys]
    return train_pairs, validation_pairs, {
        "train": train_keys,
        "validation": validation_keys,
        "test": "Reserved and not accessed by train_carer_paired.py",
    }


class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probability = torch.sigmoid(logits)
        dimensions = (1, 2, 3)
        intersection = (probability * targets).sum(dim=dimensions)
        denominator = probability.sum(dim=dimensions) + targets.sum(dim=dimensions)
        return (1.0 - (2.0 * intersection + self.eps) / (denominator + self.eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


def weighted_error_loss(prediction, target):
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.1)
    weight = 1.0 + 4.0 * target
    return (elementwise * weight).sum() / weight.sum().clamp_min(1.0)


def dice_values(logits, targets, eps=1e-6):
    probability = torch.sigmoid(logits)
    prediction = probability > 0.5
    dimensions = (1, 2, 3)
    hard_intersection = (prediction * targets.bool()).sum(dim=dimensions).float()
    hard_denominator = prediction.sum(dim=dimensions).float() + targets.sum(
        dim=dimensions
    )
    soft_intersection = (probability * targets).sum(dim=dimensions)
    soft_denominator = probability.sum(dim=dimensions) + targets.sum(dim=dimensions)
    hard = (2.0 * hard_intersection + eps) / (hard_denominator + eps)
    soft = (2.0 * soft_intersection + eps) / (soft_denominator + eps)
    return hard, soft


def make_loader(dataset, batch_size, workers, seed, shuffle):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def load_gn_weights(model, checkpoint, candidate):
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Initial checkpoint must be a training checkpoint with state_dict")
    if checkpoint.get("dataset") not in (None, "FIVES"):
        raise ValueError(f"Expected a FIVES checkpoint, found {checkpoint.get('dataset')}")
    if checkpoint.get("model_name") not in (None, "res_attention_unet_gn"):
        raise ValueError(
            "Expected model_name=res_attention_unet_gn, found "
            f"{checkpoint.get('model_name')}"
        )
    config = checkpoint.get("config") or {}
    if config.get("image_size", 512) != 512:
        raise ValueError("The paired screen requires the 512-pixel GN checkpoint")
    if config.get("input_mode", "grayscale") != "grayscale":
        raise ValueError("The paired screen requires the grayscale GN checkpoint")

    if not candidate:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return []

    incompatibility = model.load_state_dict(checkpoint["state_dict"], strict=False)
    unexpected = list(incompatibility.unexpected_keys)
    missing = list(incompatibility.missing_keys)
    allowed_prefixes = ("color_detail_stem.", "error_refiner.")
    invalid_missing = [
        key for key in missing if not key.startswith(allowed_prefixes)
    ]
    expected_new = [
        key for key in model.state_dict() if key.startswith(allowed_prefixes)
    ]
    if unexpected or invalid_missing or set(missing) != set(expected_new):
        raise RuntimeError(
            "Unsafe partial checkpoint load; "
            f"unexpected={unexpected}, invalid_missing={invalid_missing}, "
            f"missing_new_count={len(missing)}, expected_new_count={len(expected_new)}"
        )
    return missing


def verify_initial_equivalence(control, candidate, sample):
    control.eval()
    candidate.eval()
    with torch.inference_mode():
        control_logits = control(sample)
        candidate_logits = candidate(sample)
    maximum_difference = (control_logits - candidate_logits).abs().max().item()
    if maximum_difference > 1e-6:
        raise RuntimeError(
            "CARER is not baseline preserving at initialization: "
            f"max_abs_logit_difference={maximum_difference:.9g}"
        )
    return maximum_difference


def configure_trainable_parameters(model, candidate, phase):
    for parameter in model.parameters():
        parameter.requires_grad = False
    if candidate:
        for module in (model.color_detail_stem, model.error_refiner):
            for parameter in module.parameters():
                parameter.requires_grad = True
    if phase == "tail":
        for module in (model.d1, model.out_conv):
            for parameter in module.parameters():
                parameter.requires_grad = True


def build_optimizer(model, candidate, head_lr, tail_lr):
    tail_parameters = list(model.d1.parameters()) + list(model.out_conv.parameters())
    if not candidate:
        return torch.optim.Adam(tail_parameters, lr=tail_lr)
    new_parameters = list(model.color_detail_stem.parameters()) + list(
        model.error_refiner.parameters()
    )
    return torch.optim.Adam(
        [
            {"params": tail_parameters, "lr": tail_lr, "name": "gn_tail"},
            {"params": new_parameters, "lr": head_lr, "name": "carer"},
        ]
    )


def validation_metrics(model, loader, criterion, device, candidate, include_rows=False):
    model.eval()
    loss_sum = 0.0
    hard_values = []
    soft_values = []
    rows = []
    true_positive = false_positive = true_negative = false_negative = 0
    raw_hard_values = []
    raw_soft_values = []
    with torch.inference_mode():
        for images, masks, keys in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if candidate:
                logits, auxiliary = model(images, return_aux=True)
                raw_logits = auxiliary["base_logits"]
                raw_hard, raw_soft = dice_values(raw_logits, masks)
                raw_hard_values.extend(raw_hard.cpu().tolist())
                raw_soft_values.extend(raw_soft.cpu().tolist())
            else:
                logits = model(images)
            loss = criterion(logits, masks)
            loss_sum += loss.item() * images.size(0)
            hard, soft = dice_values(logits, masks)
            hard_values.extend(hard.cpu().tolist())
            soft_values.extend(soft.cpu().tolist())

            probability = torch.sigmoid(logits)
            prediction = probability > 0.5
            target = masks > 0.5
            fov = images[:, :1] > (5.0 / 255.0)
            true_positive += int((prediction & target & fov).sum().item())
            false_positive += int((prediction & ~target & fov).sum().item())
            true_negative += int((~prediction & ~target & fov).sum().item())
            false_negative += int((~prediction & target & fov).sum().item())

            if include_rows:
                for index, key in enumerate(keys):
                    fov_i = fov[index]
                    prediction_i = prediction[index]
                    target_i = target[index]
                    tp = int((prediction_i & target_i & fov_i).sum().item())
                    fp = int((prediction_i & ~target_i & fov_i).sum().item())
                    tn = int((~prediction_i & ~target_i & fov_i).sum().item())
                    fn = int((~prediction_i & target_i & fov_i).sum().item())
                    rows.append(
                        {
                            "key": key,
                            "hard_dice": float(hard[index].item()),
                            "soft_dice": float(soft[index].item()),
                            "sensitivity": tp / max(1, tp + fn),
                            "specificity": tn / max(1, tn + fp),
                            "tp": tp,
                            "fp": fp,
                            "tn": tn,
                            "fn": fn,
                        }
                    )

    sensitivity = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    result = {
        "validation_loss": loss_sum / len(loader.dataset),
        "validation_dice": float(np.mean(hard_values)),
        "validation_soft_dice": float(np.mean(soft_values)),
        "validation_sensitivity": sensitivity,
        "validation_specificity": specificity,
    }
    if candidate:
        result["raw_validation_dice"] = float(np.mean(raw_hard_values))
        result["raw_validation_soft_dice"] = float(np.mean(raw_soft_values))
    return result, rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, model_name, epoch, metrics, config):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "dataset": "FIVES",
            "epoch": epoch,
            "validation_dice": metrics["validation_dice"],
            "validation_soft_dice": metrics["validation_soft_dice"],
            "validation_loss": metrics["validation_loss"],
            "selection_metric": "soft_dice",
            "selection_value": metrics["validation_soft_dice"],
            "input_mode": "gray_rgb",
            "config": config,
        },
        path,
    )


def train_variant(
    model,
    model_name,
    train_dataset,
    validation_dataset,
    split_record,
    args,
    device,
    candidate,
    checkpoint_hash,
    initial_equivalence,
):
    run_dir = args.run_root / "FIVES" / model_name / f"seed_{args.train_seed}"
    result_path = run_dir / "best_result.json"
    log_path = run_dir / "training_log.csv"
    per_image_path = run_dir / "validation_per_image.csv"
    checkpoint_path = args.weight_dir / f"FIVES_best_{model_name}_seed{args.train_seed}.pth"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"Completed run already exists: {result_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    args.weight_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": "FIVES",
        "model": model_name,
        "data_root": str(args.data_root.resolve()),
        "run_root": str(args.run_root.resolve()),
        "weight_dir": str(args.weight_dir.resolve()),
        "image_size": args.image_size,
        "input_mode": "gray_rgb",
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs if candidate else 0,
        "tail_start_epoch": args.warmup_epochs + 1,
        "batch_size": args.batch_size,
        "head_learning_rate": args.head_learning_rate if candidate else None,
        "tail_learning_rate": args.tail_learning_rate,
        "base_loss_weight": args.base_loss_weight if candidate else 0.0,
        "error_loss_weight": args.error_loss_weight if candidate else 0.0,
        "train_seed": args.train_seed,
        "workers": args.workers,
        "selection_metric": "soft_dice",
        "metric_threshold": 0.5,
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "initial_checkpoint_sha256": checkpoint_hash,
        "split_file": str(args.split_file.resolve()),
        "parameter_count": count_parameters(model),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "device_resolved": str(device),
        "initial_max_abs_logit_difference": initial_equivalence,
        "official_test_accessed": False,
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "split.json", split_record)

    criterion = BCEDiceLoss()
    configure_trainable_parameters(model, candidate, phase="head" if candidate else "tail")
    optimizer = build_optimizer(
        model, candidate, args.head_learning_rate, args.tail_learning_rate
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, args.workers, args.train_seed + 900000, False
    )

    initial_metrics, _ = validation_metrics(
        model, validation_loader, criterion, device, candidate
    )
    best_metrics = dict(initial_metrics)
    best_epoch = 0
    best_score = initial_metrics["validation_soft_dice"]
    save_checkpoint(
        checkpoint_path, model, model_name, 0, initial_metrics, config
    )
    rows = [
        {
            "epoch": 0,
            "phase": "initial",
            "train_total_loss": "",
            "train_final_loss": "",
            "train_base_loss": "",
            "train_fn_loss": "",
            "train_fp_loss": "",
            "mean_abs_input_delta": 0.0,
            "mean_abs_logit_delta": 0.0,
            **initial_metrics,
            "tail_learning_rate": args.tail_learning_rate,
            "head_learning_rate": args.head_learning_rate if candidate else 0.0,
            "is_best": 1,
        }
    ]
    print(
        f"{model_name} epoch 0: val_loss={initial_metrics['validation_loss']:.6f}, "
        f"hard_dice={initial_metrics['validation_dice']:.6f}, "
        f"soft_dice={initial_metrics['validation_soft_dice']:.6f}"
    )

    first_epoch = 1 if candidate else args.warmup_epochs + 1
    for epoch in range(first_epoch, args.epochs + 1):
        phase = "head_warmup" if candidate and epoch <= args.warmup_epochs else "tail"
        configure_trainable_parameters(model, candidate, phase)
        model.train()
        epoch_seed = args.train_seed * 10000 + epoch
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % (2**32))
        torch.manual_seed(epoch_seed)
        train_loader = make_loader(
            train_dataset, args.batch_size, args.workers, epoch_seed, True
        )

        sums = {
            "total": 0.0,
            "final": 0.0,
            "base": 0.0,
            "fn": 0.0,
            "fp": 0.0,
            "input_delta": 0.0,
            "logit_delta": 0.0,
        }
        progress = tqdm(
            train_loader,
            desc=f"{model_name} epoch {epoch}/{args.epochs} {phase}",
            ncols=110,
        )
        for images, masks, _ in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if candidate:
                logits, auxiliary = model(images, return_aux=True)
                final_loss = criterion(logits, masks)
                base_loss = criterion(auxiliary["base_logits"], masks)
                base_probability = torch.sigmoid(auxiliary["base_logits"].detach())
                fn_target = masks * (1.0 - base_probability)
                fp_target = (1.0 - masks) * base_probability
                fn_loss = weighted_error_loss(auxiliary["fn_correction"], fn_target)
                fp_loss = weighted_error_loss(auxiliary["fp_correction"], fp_target)
                protected_main = (
                    final_loss + args.base_loss_weight * base_loss
                ) / (1.0 + args.base_loss_weight)
                loss = protected_main + args.error_loss_weight * 0.5 * (
                    fn_loss + fp_loss
                )
                input_delta = auxiliary["input_delta"].abs().mean()
                logit_delta = auxiliary["logit_delta"].abs().mean()
            else:
                logits = model(images)
                final_loss = criterion(logits, masks)
                base_loss = final_loss
                fn_loss = fp_loss = final_loss.detach().new_zeros(())
                input_delta = logit_delta = final_loss.detach().new_zeros(())
                loss = final_loss

            loss.backward()
            optimizer.step()
            batch_count = images.size(0)
            sums["total"] += loss.item() * batch_count
            sums["final"] += final_loss.item() * batch_count
            sums["base"] += base_loss.item() * batch_count
            sums["fn"] += fn_loss.item() * batch_count
            sums["fp"] += fp_loss.item() * batch_count
            sums["input_delta"] += input_delta.item() * batch_count
            sums["logit_delta"] += logit_delta.item() * batch_count
            progress.set_postfix(loss=f"{loss.item():.4f}")

        validation_loader = make_loader(
            validation_dataset,
            args.batch_size,
            args.workers,
            args.train_seed + 900000,
            False,
        )
        metrics, _ = validation_metrics(
            model, validation_loader, criterion, device, candidate
        )
        improved = metrics["validation_soft_dice"] > best_score
        if improved:
            best_score = metrics["validation_soft_dice"]
            best_epoch = epoch
            best_metrics = dict(metrics)
            save_checkpoint(
                checkpoint_path, model, model_name, epoch, metrics, config
            )
        count = len(train_dataset)
        rows.append(
            {
                "epoch": epoch,
                "phase": phase,
                "train_total_loss": sums["total"] / count,
                "train_final_loss": sums["final"] / count,
                "train_base_loss": sums["base"] / count,
                "train_fn_loss": sums["fn"] / count,
                "train_fp_loss": sums["fp"] / count,
                "mean_abs_input_delta": sums["input_delta"] / count,
                "mean_abs_logit_delta": sums["logit_delta"] / count,
                **metrics,
                "tail_learning_rate": args.tail_learning_rate,
                "head_learning_rate": args.head_learning_rate if candidate else 0.0,
                "is_best": int(improved),
            }
        )
        write_csv(log_path, rows)
        print(
            f"{model_name} epoch {epoch}: val_loss={metrics['validation_loss']:.6f}, "
            f"hard_dice={metrics['validation_dice']:.6f}, "
            f"soft_dice={metrics['validation_soft_dice']:.6f}, "
            f"best={best_score:.6f} (epoch {best_epoch})"
        )

    best_checkpoint = load_torch_checkpoint(checkpoint_path)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    validation_loader = make_loader(
        validation_dataset, args.batch_size, args.workers, args.train_seed + 900000, False
    )
    confirmed_metrics, per_image_rows = validation_metrics(
        model, validation_loader, criterion, device, candidate, include_rows=True
    )
    write_csv(per_image_path, per_image_rows)
    best_metrics.update(confirmed_metrics)
    result = {
        "best_epoch": best_epoch,
        "selection_metric": "soft_dice",
        "best_selection_value": best_score,
        "best_validation_loss": best_metrics["validation_loss"],
        "best_validation_dice": best_metrics["validation_dice"],
        "best_validation_soft_dice": best_metrics["validation_soft_dice"],
        "best_validation_sensitivity": best_metrics["validation_sensitivity"],
        "best_validation_specificity": best_metrics["validation_specificity"],
        "checkpoint": str(checkpoint_path.resolve()),
        "official_test_accessed": False,
    }
    if candidate:
        result["raw_validation_dice_at_best"] = best_metrics["raw_validation_dice"]
        result["raw_validation_soft_dice_at_best"] = best_metrics[
            "raw_validation_soft_dice"
        ]
    write_json(result_path, result)
    print(f"Completed {model_name}; best checkpoint: {checkpoint_path}")
    return result


def main():
    args = parse_args()
    if args.epochs <= args.warmup_epochs:
        raise ValueError("epochs must be greater than warmup-epochs")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.base_loss_weight < 0 or args.error_loss_weight < 0:
        raise ValueError("loss weights cannot be negative")
    device = resolve_device(args.device)
    set_seed(args.train_seed)

    all_pairs = discover_pairs(args.data_root, "FIVES", "train")
    train_pairs, validation_pairs, split_record = load_explicit_split(
        all_pairs, args.split_file
    )
    train_dataset = RetinalDataset(
        train_pairs, args.image_size, input_mode="gray_rgb", augment=True
    )
    validation_dataset = RetinalDataset(
        validation_pairs, args.image_size, input_mode="gray_rgb", augment=False
    )
    checkpoint = load_torch_checkpoint(args.initial_checkpoint)
    checkpoint_hash = file_sha256(args.initial_checkpoint)

    set_seed(args.train_seed)
    control = GrayRGBControlUNet()
    load_gn_weights(control, checkpoint, candidate=False)
    set_seed(args.train_seed)
    candidate = CARERUNet()
    missing_new_keys = load_gn_weights(candidate, checkpoint, candidate=True)

    equivalence_loader = make_loader(
        validation_dataset, 1, 0, args.train_seed + 700000, False
    )
    sample, _, _ = next(iter(equivalence_loader))
    # A smaller crop makes the safety check fast on CPU while exercising every
    # stage of the encoder/decoder. The full 512 validation follows immediately.
    sample = sample[:, :, :128, :128]
    initial_equivalence = verify_initial_equivalence(control, candidate, sample)

    print(f"Device: {device}")
    print(f"Explicit split: train={len(train_pairs)}, validation={len(validation_pairs)}")
    print(f"Initial checkpoint SHA256: {checkpoint_hash}")
    print(f"Allowed new CARER tensors initialized: {len(missing_new_keys)}")
    print(f"Initial control/CARER max abs logit difference: {initial_equivalence:.9g}")
    print("Official FIVES test set is reserved and was not discovered or accessed.")

    control = control.to(device)
    control_result = train_variant(
        control,
        MODEL_NAMES[0],
        train_dataset,
        validation_dataset,
        split_record,
        args,
        device,
        candidate=False,
        checkpoint_hash=checkpoint_hash,
        initial_equivalence=initial_equivalence,
    )
    control.cpu()
    del control
    if device.type == "cuda":
        torch.cuda.empty_cache()

    candidate = candidate.to(device)
    candidate_result = train_variant(
        candidate,
        MODEL_NAMES[1],
        train_dataset,
        validation_dataset,
        split_record,
        args,
        device,
        candidate=True,
        checkpoint_hash=checkpoint_hash,
        initial_equivalence=initial_equivalence,
    )
    print(
        "Paired training complete: "
        f"control_soft={control_result['best_validation_soft_dice']:.6f}, "
        f"carer_soft={candidate_result['best_validation_soft_dice']:.6f}"
    )


if __name__ == "__main__":
    main()
