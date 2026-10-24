import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    DATASET_CHOICES,
    RetinalDataset,
    discover_pairs,
    seed_worker,
    split_train_validation,
)
from models.model_factory import MODEL_CHOICES, build_model, checkpoint_tag, count_parameters


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train retinal vessel models reproducibly.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="CHASE")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="improved_attention_unet")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--weight-dir", type=Path, default=PROJECT_ROOT / "weight_files")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--input-mode", choices=("grayscale", "green", "gray_rgb"), default="grayscale"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--loss",
        choices=(
            "bce",
            "bce_dice",
            "bce_dice_connectivity",
            "bce_dice_connectivity_annealed",
        ),
        default="bce",
        help=(
            "Training loss. bce_dice_connectivity adds training-only centerline "
            "supervision and requires connectivity_aux_unet."
        ),
    )
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument(
        "--centerline-weight",
        type=float,
        default=0.15,
        help="Weight of the auxiliary centerline BCE+Dice loss.",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.05,
        help="Weight of the centerline/segmentation consistency loss.",
    )
    parser.add_argument(
        "--skeleton-iterations",
        type=int,
        default=15,
        help="Differentiable erosion iterations used to derive centerlines.",
    )
    parser.add_argument(
        "--aux-anneal-start-epoch",
        type=int,
        default=10,
        help="Last epoch that uses the full auxiliary weights.",
    )
    parser.add_argument(
        "--aux-anneal-end-epoch",
        type=int,
        default=30,
        help="Epoch at which cosine-annealed auxiliary weights reach zero.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-file",
        type=Path,
        help=(
            "Optional explicit train/validation split JSON. When supplied, the "
            "file must exactly cover the prepared training pool."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=30,
        help="Do not early-stop before this epoch.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("soft_dice", "hard_dice", "validation_loss"),
        default="soft_dice",
        help="Metric used for checkpoint selection, LR scheduling, and early stopping.",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def dice_per_image(logits, targets, threshold=0.5, eps=1e-6):
    predictions = (torch.sigmoid(logits) > threshold).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    denominator = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (2 * intersection + eps) / (denominator + eps)


def soft_dice_per_image(logits, targets, eps=1e-6):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (2 * intersection + eps) / (denominator + eps)


class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        dimensions = (1, 2, 3)
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        return (1 - dice).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        total_weight = bce_weight + dice_weight
        self.bce_weight = bce_weight / total_weight
        self.dice_weight = dice_weight / total_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


def soft_erode(mask):
    """Differentiable 2-D erosion used by soft skeletonization."""
    vertical = -torch.nn.functional.max_pool2d(
        -mask, kernel_size=(3, 1), stride=1, padding=(1, 0)
    )
    horizontal = -torch.nn.functional.max_pool2d(
        -mask, kernel_size=(1, 3), stride=1, padding=(0, 1)
    )
    return torch.minimum(vertical, horizontal)


def soft_skeletonize(mask, iterations=15):
    """Approximate a centerline while preserving gradients for predictions."""
    if iterations < 1:
        raise ValueError("skeleton_iterations must be at least 1")

    def soft_open(value):
        eroded = soft_erode(value)
        return torch.nn.functional.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)

    opened = soft_open(mask)
    skeleton = torch.relu(mask - opened)
    working = mask
    for _ in range(iterations):
        working = soft_erode(working)
        opened = soft_open(working)
        delta = torch.relu(working - opened)
        skeleton = skeleton + torch.relu(delta - skeleton * delta)
    return skeleton.clamp_(0.0, 1.0)


class ConnectivityAuxiliaryLoss(nn.Module):
    """Segmentation loss plus centerline supervision and bidirectional consistency."""

    requires_auxiliary_output = True

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        centerline_weight=0.15,
        consistency_weight=0.05,
        skeleton_iterations=15,
        anneal_start_epoch=None,
        anneal_end_epoch=None,
        eps=1e-6,
    ):
        super().__init__()
        if centerline_weight < 0 or consistency_weight < 0:
            raise ValueError("Auxiliary loss weights must be non-negative")
        if skeleton_iterations < 1:
            raise ValueError("skeleton_iterations must be at least 1")
        if (anneal_start_epoch is None) != (anneal_end_epoch is None):
            raise ValueError("Both auxiliary annealing epochs must be set together")
        if anneal_start_epoch is not None:
            if anneal_start_epoch < 1:
                raise ValueError("aux_anneal_start_epoch must be at least 1")
            if anneal_end_epoch <= anneal_start_epoch:
                raise ValueError(
                    "aux_anneal_end_epoch must be greater than aux_anneal_start_epoch"
                )
        self.main_loss = BCEDiceLoss(bce_weight, dice_weight)
        self.centerline_loss = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
        self.centerline_weight = centerline_weight
        self.consistency_weight = consistency_weight
        self.skeleton_iterations = skeleton_iterations
        self.anneal_start_epoch = anneal_start_epoch
        self.anneal_end_epoch = anneal_end_epoch
        self.current_epoch = 1
        self.eps = eps

    def set_epoch(self, epoch):
        if epoch < 1:
            raise ValueError("epoch must be at least 1")
        self.current_epoch = epoch

    @property
    def auxiliary_scale(self):
        if self.anneal_start_epoch is None:
            return 1.0
        if self.current_epoch <= self.anneal_start_epoch:
            return 1.0
        if self.current_epoch >= self.anneal_end_epoch:
            return 0.0
        progress = (
            (self.current_epoch - self.anneal_start_epoch)
            / (self.anneal_end_epoch - self.anneal_start_epoch)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _centerline_target(self, targets):
        with torch.no_grad():
            return soft_skeletonize(targets, self.skeleton_iterations)

    def forward(self, logits, centerline_logits, targets):
        auxiliary_scale = self.auxiliary_scale
        if auxiliary_scale <= 0.0:
            main_loss = self.main_loss(logits, targets)
            zero = main_loss.detach().new_zeros(())
            return main_loss, {
                "main_loss": main_loss.detach(),
                "centerline_loss": zero,
                "consistency_loss": zero,
                "auxiliary_scale": zero,
            }

        main_loss = self.main_loss(logits, targets)
        centerline_target = self._centerline_target(targets)
        centerline_loss = self.centerline_loss(centerline_logits, centerline_target)

        dimensions = (1, 2, 3)
        main_probability = torch.sigmoid(logits)
        centerline_probability = torch.sigmoid(centerline_logits)

        # Ground-truth centerlines must be covered by the main segmentation.
        target_mass = centerline_target.sum(dim=dimensions)
        coverage = 1.0 - (
            (main_probability * centerline_target).sum(dim=dimensions) + self.eps
        ) / (target_mass + self.eps)

        # Auxiliary centerlines should lie inside both the annotation and the
        # main prediction, suppressing disconnected false-positive branches.
        predicted_mass = centerline_probability.sum(dim=dimensions)
        outside_target = (
            centerline_probability * (1.0 - targets)
        ).sum(dim=dimensions) / (predicted_mass + self.eps)
        outside_main = (
            centerline_probability * (1.0 - main_probability)
        ).sum(dim=dimensions) / (predicted_mass + self.eps)
        consistency_loss = (coverage + outside_target + outside_main).mean() / 3.0

        total_loss = (
            main_loss
            + auxiliary_scale * self.centerline_weight * centerline_loss
            + auxiliary_scale * self.consistency_weight * consistency_loss
        )
        components = {
            "main_loss": main_loss.detach(),
            "centerline_loss": centerline_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "auxiliary_scale": main_loss.detach().new_tensor(auxiliary_scale),
        }
        return total_loss, components

    def validation_loss(self, logits, targets):
        """Keep validation loss directly comparable with the GN baseline."""
        return self.main_loss(logits, targets)


def build_criterion(
    name,
    bce_weight=1.0,
    dice_weight=1.0,
    centerline_weight=0.15,
    consistency_weight=0.05,
    skeleton_iterations=15,
    aux_anneal_start_epoch=10,
    aux_anneal_end_epoch=30,
):
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "bce_dice":
        return BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    if name == "bce_dice_connectivity":
        return ConnectivityAuxiliaryLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            centerline_weight=centerline_weight,
            consistency_weight=consistency_weight,
            skeleton_iterations=skeleton_iterations,
        )
    if name == "bce_dice_connectivity_annealed":
        return ConnectivityAuxiliaryLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            centerline_weight=centerline_weight,
            consistency_weight=consistency_weight,
            skeleton_iterations=skeleton_iterations,
            anneal_start_epoch=aux_anneal_start_epoch,
            anneal_end_epoch=aux_anneal_end_epoch,
        )
    raise ValueError(f"Unknown loss: {name}")


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    hard_dice_values = []
    soft_dice_values = []
    with torch.inference_mode():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            if hasattr(criterion, "validation_loss"):
                validation_loss = criterion.validation_loss(logits, masks)
            else:
                validation_loss = criterion(logits, masks)
            loss_sum += validation_loss.item() * images.size(0)
            hard_dice_values.extend(dice_per_image(logits, masks).cpu().tolist())
            soft_dice_values.extend(soft_dice_per_image(logits, masks).cpu().tolist())
    return (
        loss_sum / len(loader.dataset),
        float(np.mean(hard_dice_values)),
        float(np.mean(soft_dice_values)),
    )


def selection_value(name, validation_loss, hard_dice, soft_dice):
    if name == "validation_loss":
        return validation_loss
    if name == "hard_dice":
        return hard_dice
    return soft_dice


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def load_explicit_split(all_pairs, split_file):
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
            "Explicit split does not exactly cover prepared training data; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return (
        [by_key[key] for key in train_keys],
        [by_key[key] for key in validation_keys],
    )


def main():
    args = parse_args()
    connectivity_losses = {
        "bce_dice_connectivity",
        "bce_dice_connectivity_annealed",
    }
    if args.loss in connectivity_losses and args.model != "connectivity_aux_unet":
        raise ValueError(
            "Connectivity losses require --model connectivity_aux_unet"
        )
    if args.model == "connectivity_aux_unet" and args.loss not in connectivity_losses:
        raise ValueError(
            "connectivity_aux_unet must be trained with a connectivity loss"
        )
    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)

    run_dir = args.run_root / args.dataset / args.model / f"seed_{args.seed}"
    log_path = run_dir / "training_log.csv"
    if log_path.exists() and not args.overwrite:
        raise FileExistsError(f"Run already exists: {run_dir}. Pass --overwrite to replace its logs.")
    run_dir.mkdir(parents=True, exist_ok=True)
    args.weight_dir.mkdir(parents=True, exist_ok=True)

    all_train_pairs = discover_pairs(args.data_root, args.dataset, "train")
    if args.split_file is None:
        train_pairs, validation_pairs = split_train_validation(
            all_train_pairs, args.validation_ratio, args.seed
        )
    else:
        train_pairs, validation_pairs = load_explicit_split(
            all_train_pairs, args.split_file
        )
    train_dataset = RetinalDataset(
        train_pairs, args.image_size, args.input_mode, augment=not args.no_augmentation
    )
    validation_dataset = RetinalDataset(
        validation_pairs, args.image_size, args.input_mode, augment=False
    )

    generator = torch.Generator().manual_seed(args.seed)
    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common_loader_args)

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
    criterion = build_criterion(
        args.loss,
        args.bce_weight,
        args.dice_weight,
        args.centerline_weight,
        args.consistency_weight,
        args.skeleton_iterations,
        args.aux_anneal_start_epoch,
        args.aux_anneal_end_epoch,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    selection_mode = "min" if args.selection_metric == "validation_loss" else "max"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=selection_mode, factor=0.5, patience=5
    )

    tag = checkpoint_tag(args.model)
    checkpoint_path = args.weight_dir / f"{args.dataset}_best_{tag}_seed{args.seed}.pth"
    configuration = vars(args).copy()
    configuration.update(
        {
            "data_root": str(args.data_root.resolve()),
            "run_root": str(args.run_root.resolve()),
            "weight_dir": str(args.weight_dir.resolve()),
            "device_resolved": str(device),
            "parameter_count": count_parameters(model),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "checkpoint": str(checkpoint_path.resolve()),
            "split_file": (
                str(args.split_file.resolve()) if args.split_file is not None else None
            ),
        }
    )
    write_json(run_dir / "config.json", configuration)
    write_json(
        run_dir / "split.json",
        {
            "train": [pair[2] for pair in train_pairs],
            "validation": [pair[2] for pair in validation_pairs],
            "test": "Reserved and not accessed by train.py",
            "source": (
                str(args.split_file.resolve()) if args.split_file is not None else "generated"
            ),
        },
    )

    best_score = float("inf") if selection_mode == "min" else float("-inf")
    best_epoch = 0
    best_validation_loss = None
    best_validation_dice = None
    best_validation_soft_dice = None
    epochs_without_improvement = 0
    rows = []

    print(f"Dataset: {args.dataset} | model: {args.model} | device: {device}")
    print(f"Train/validation samples: {len(train_dataset)}/{len(validation_dataset)}")
    print(f"Parameters: {count_parameters(model):,}")
    print(
        f"Loss: {args.loss} | BCE weight: {args.bce_weight:g} | "
        f"Dice weight: {args.dice_weight:g}"
    )
    if args.loss in connectivity_losses:
        print(
            f"Centerline weight: {args.centerline_weight:g} | "
            f"consistency weight: {args.consistency_weight:g} | "
            f"skeleton iterations: {args.skeleton_iterations}"
        )
    if args.loss == "bce_dice_connectivity_annealed":
        print(
            f"Auxiliary cosine annealing: full through epoch "
            f"{args.aux_anneal_start_epoch}, zero at epoch "
            f"{args.aux_anneal_end_epoch}"
        )
    print(
        f"Selection metric: {args.selection_metric} | min epochs: {args.min_epochs} | "
        f"patience: {args.patience}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        epoch_auxiliary_scale = float(getattr(criterion, "auxiliary_scale", 0.0))
        running_loss = 0.0
        running_main_loss = 0.0
        running_centerline_loss = 0.0
        running_consistency_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for images, masks, _ in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if (
                getattr(criterion, "requires_auxiliary_output", False)
                and epoch_auxiliary_scale > 0.0
            ):
                logits, centerline_logits = model(images, return_aux=True)
                loss, components = criterion(logits, centerline_logits, masks)
            elif getattr(criterion, "requires_auxiliary_output", False):
                # Once annealing reaches zero, skip the auxiliary head and all
                # skeleton operations so late training has baseline GN cost.
                logits = model(images)
                loss = criterion.validation_loss(logits, masks)
                components = {
                    "main_loss": loss.detach(),
                    "centerline_loss": torch.zeros((), device=device),
                    "consistency_loss": torch.zeros((), device=device),
                    "auxiliary_scale": torch.zeros((), device=device),
                }
            else:
                logits = model(images)
                loss = criterion(logits, masks)
                components = {
                    "main_loss": loss.detach(),
                    "centerline_loss": torch.zeros((), device=device),
                    "consistency_loss": torch.zeros((), device=device),
                    "auxiliary_scale": torch.zeros((), device=device),
                }
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_main_loss += components["main_loss"].item() * images.size(0)
            running_centerline_loss += (
                components["centerline_loss"].item() * images.size(0)
            )
            running_consistency_loss += (
                components["consistency_loss"].item() * images.size(0)
            )
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)
        train_main_loss = running_main_loss / len(train_loader.dataset)
        train_centerline_loss = running_centerline_loss / len(train_loader.dataset)
        train_consistency_loss = running_consistency_loss / len(train_loader.dataset)
        validation_loss, validation_dice, validation_soft_dice = validate(
            model, validation_loader, criterion, device
        )
        score = selection_value(
            args.selection_metric,
            validation_loss,
            validation_dice,
            validation_soft_dice,
        )
        scheduler.step(score)
        learning_rate = optimizer.param_groups[0]["lr"]
        improved = score < best_score if selection_mode == "min" else score > best_score

        if improved:
            best_score = score
            best_epoch = epoch
            best_validation_loss = validation_loss
            best_validation_dice = validation_dice
            best_validation_soft_dice = validation_soft_dice
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": args.model,
                    "dataset": args.dataset,
                    "epoch": epoch,
                    "validation_dice": validation_dice,
                    "validation_soft_dice": validation_soft_dice,
                    "validation_loss": validation_loss,
                    "selection_metric": args.selection_metric,
                    "selection_value": score,
                    "config": configuration,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_main_loss": train_main_loss,
                "train_centerline_loss": train_centerline_loss,
                "train_consistency_loss": train_consistency_loss,
                "auxiliary_scale": epoch_auxiliary_scale,
                "effective_centerline_weight": (
                    args.centerline_weight * epoch_auxiliary_scale
                ),
                "effective_consistency_weight": (
                    args.consistency_weight * epoch_auxiliary_scale
                ),
                "validation_loss": validation_loss,
                "validation_dice": validation_dice,
                "validation_soft_dice": validation_soft_dice,
                "selection_value": score,
                "learning_rate": learning_rate,
                "is_best": int(improved),
            }
        )
        with log_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={validation_loss:.4f}, "
            f"val_dice={validation_dice:.4f}, val_soft_dice={validation_soft_dice:.4f}, "
            f"best_{args.selection_metric}={best_score:.4f} (epoch {best_epoch})"
        )
        if args.loss in connectivity_losses:
            print(
                f"  components: main={train_main_loss:.4f}, "
                f"centerline={train_centerline_loss:.4f}, "
                f"consistency={train_consistency_loss:.4f}, "
                f"aux_scale={epoch_auxiliary_scale:.4f}"
            )
        if (
            args.patience > 0
            and epoch >= args.min_epochs
            and epochs_without_improvement >= args.patience
        ):
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    write_json(
        run_dir / "best_result.json",
        {
            "best_epoch": best_epoch,
            "selection_metric": args.selection_metric,
            "best_selection_value": best_score,
            "best_validation_loss": best_validation_loss,
            "best_validation_dice": best_validation_dice,
            "best_validation_soft_dice": best_validation_soft_dice,
            "checkpoint": str(checkpoint_path.resolve()),
        },
    )
    print(f"Training complete. Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
