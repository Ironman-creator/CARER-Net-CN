import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_utils import DATASET_CHOICES, RetinalDataset, discover_pairs
from models.Attention_UNet_V5 import (
    CARERUNet,
    ContrastOnlyColorDetailUNet,
    ColorDetailOnlyUNet,
    GrayCLAHEUNet,
    GrayRGBControlUNet,
    GreenChannelUNet,
    LearnedRGBProjectionUNet,
    RGBOnlyColorDetailUNet,
    SignedRefinerOnlyUNet,
)
from models.model_factory import count_parameters
from train_carer_paired import (
    BCEDiceLoss,
    file_sha256,
    load_explicit_split,
    load_torch_checkpoint,
    make_loader,
    resolve_device,
    set_seed,
    validation_metrics,
    weighted_error_loss,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent
VARIANTS = (
    "gray_rgb_control_30_unet",
    "color_detail_only_unet",
    "signed_refiner_only_unet",
)
MODEL_BUILDERS = {
    "gray_rgb_control_30_unet": GrayRGBControlUNet,
    "color_detail_only_unet": ColorDetailOnlyUNet,
    "signed_refiner_only_unet": SignedRefinerOnlyUNet,
    "carer_unet": CARERUNet,
    "green_channel_unet": GreenChannelUNet,
    "gray_clahe_unet": GrayCLAHEUNet,
    "rgb_1x1_projection_unet": LearnedRGBProjectionUNet,
    "carer_rgb_only_unet": RGBOnlyColorDetailUNet,
    "carer_contrast_only_unet": ContrastOnlyColorDetailUNet,
}

INPUT_MODES = {
    "gray_rgb_control_30_unet": "gray_rgb",
    "color_detail_only_unet": "gray_rgb",
    "signed_refiner_only_unet": "gray_rgb",
    "carer_unet": "gray_rgb",
    "green_channel_unet": "gray_rgb",
    "gray_clahe_unet": "gray_rgb_clahe",
    "rgb_1x1_projection_unet": "gray_rgb",
    "carer_rgb_only_unet": "gray_rgb",
    "carer_contrast_only_unet": "gray_rgb",
}

MODULE_NAMES = {
    "color_detail_only_unet": ("color_detail_stem",),
    "signed_refiner_only_unet": ("error_refiner",),
    "carer_unet": ("color_detail_stem", "error_refiner"),
    "rgb_1x1_projection_unet": ("rgb_projection",),
    "carer_rgb_only_unet": ("color_detail_stem",),
    "carer_contrast_only_unet": ("color_detail_stem",),
}

RETURNS_AUX = {
    "color_detail_only_unet",
    "signed_refiner_only_unet",
    "carer_unet",
    "rgb_1x1_projection_unet",
    "carer_rgb_only_unet",
    "carer_contrast_only_unet",
}

BASELINE_PRESERVING = {
    "gray_rgb_control_30_unet",
    "color_detail_only_unet",
    "signed_refiner_only_unet",
    "carer_unet",
    "carer_rgb_only_unet",
    "carer_contrast_only_unet",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute-matched CARER ablations on an explicit retinal split."
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="FIVES")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--run-root", type=Path, default=PROJECT_ROOT / "runs_fives_carer_ablation"
    )
    parser.add_argument(
        "--weight-dir",
        type=Path,
        default=PROJECT_ROOT / "weight_files_fives_carer_ablation",
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
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(MODEL_BUILDERS),
        default=list(VARIANTS),
        help="Ordered experiment arms. Defaults to the original three-arm ablation.",
    )
    parser.add_argument(
        "--experiment-name", default="CARER compute-matched ablation"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path):
    import json

    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def variant_modules(model, model_name):
    return [getattr(model, name) for name in MODULE_NAMES.get(model_name, ())]


def load_gn_weights(model, checkpoint, model_name, dataset):
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Initial checkpoint must contain state_dict")
    if checkpoint.get("dataset") not in (None, dataset):
        raise ValueError(
            f"Expected {dataset} checkpoint, found {checkpoint.get('dataset')}"
        )
    if checkpoint.get("model_name") not in (None, "res_attention_unet_gn"):
        raise ValueError(
            "Expected model_name=res_attention_unet_gn, found "
            f"{checkpoint.get('model_name')}"
        )
    config = checkpoint.get("config") or {}
    if config.get("image_size", 512) != 512:
        raise ValueError("The ablation requires the 512-pixel GN checkpoint")
    if config.get("input_mode", "grayscale") != "grayscale":
        raise ValueError("The ablation requires the grayscale GN checkpoint")

    allowed_module_names = MODULE_NAMES.get(model_name, ())
    if not allowed_module_names:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return []

    allowed_prefixes = tuple(f"{name}." for name in allowed_module_names)
    incompatibility = model.load_state_dict(checkpoint["state_dict"], strict=False)
    missing = list(incompatibility.missing_keys)
    unexpected = list(incompatibility.unexpected_keys)
    expected = [key for key in model.state_dict() if key.startswith(allowed_prefixes)]
    if unexpected or set(missing) != set(expected):
        raise RuntimeError(
            f"Unsafe partial checkpoint load for {model_name}; "
            f"unexpected={unexpected}, missing={missing}, expected={expected}"
        )
    return missing


def measure_initial_differences(control, candidates, samples_by_mode):
    control.eval()
    with torch.inference_mode():
        differences = {}
        for model_name, model in candidates.items():
            sample = samples_by_mode[INPUT_MODES[model_name]]
            # Every multi-channel representation keeps the exact original
            # grayscale image in channel zero, so the reference FOV and GN
            # control remain identical across all five experiment arms.
            control_logits = control(sample[:, :4])
            model.eval()
            candidate_logits = model(sample)
            difference = (control_logits - candidate_logits).abs().max().item()
            if model_name in BASELINE_PRESERVING and difference > 1e-6:
                raise RuntimeError(
                    f"{model_name} is not baseline preserving: max difference={difference}"
                )
            differences[model_name] = difference
    return differences


def configure_trainable_parameters(model, model_name, phase):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in variant_modules(model, model_name):
        for parameter in module.parameters():
            parameter.requires_grad = True
    if phase == "tail":
        for module in (model.d1, model.out_conv):
            for parameter in module.parameters():
                parameter.requires_grad = True


def build_optimizer(model, model_name, head_learning_rate, tail_learning_rate):
    groups = [
        {
            "params": list(model.d1.parameters()) + list(model.out_conv.parameters()),
            "lr": tail_learning_rate,
            "name": "gn_tail",
        }
    ]
    new_parameters = [
        parameter
        for module in variant_modules(model, model_name)
        for parameter in module.parameters()
    ]
    if new_parameters:
        groups.append(
            {"params": new_parameters, "lr": head_learning_rate, "name": "new_module"}
        )
    return torch.optim.Adam(groups)


def save_checkpoint(path, model, model_name, epoch, metrics, config, dataset):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "dataset": dataset,
            "epoch": epoch,
            "validation_dice": metrics["validation_dice"],
            "validation_soft_dice": metrics["validation_soft_dice"],
            "validation_loss": metrics["validation_loss"],
            "selection_metric": "soft_dice",
            "selection_value": metrics["validation_soft_dice"],
            "input_mode": INPUT_MODES[model_name],
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
    checkpoint_hash,
    initial_equivalence,
):
    run_dir = args.run_root / args.dataset / model_name / f"seed_{args.train_seed}"
    result_path = run_dir / "best_result.json"
    log_path = run_dir / "training_log.csv"
    per_image_path = run_dir / "validation_per_image.csv"
    checkpoint_path = (
        args.weight_dir / f"{args.dataset}_best_{model_name}_seed{args.train_seed}.pth"
    )
    if result_path.exists() and not args.overwrite:
        print(f"SKIP completed arm: {result_path}")
        return read_json(result_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    args.weight_dir.mkdir(parents=True, exist_ok=True)

    has_new_module = bool(variant_modules(model, model_name))
    returns_aux = model_name in RETURNS_AUX
    has_refiner = model_name in {"signed_refiner_only_unet", "carer_unet"}
    config = {
        "dataset": args.dataset,
        "experiment": args.experiment_name,
        "model": model_name,
        "data_root": str(args.data_root.resolve()),
        "run_root": str(args.run_root.resolve()),
        "weight_dir": str(args.weight_dir.resolve()),
        "image_size": args.image_size,
        "input_mode": INPUT_MODES[model_name],
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs if has_new_module else 0,
        "tail_start_epoch": args.warmup_epochs + 1 if has_new_module else 1,
        "batch_size": args.batch_size,
        "head_learning_rate": args.head_learning_rate if has_new_module else None,
        "tail_learning_rate": args.tail_learning_rate,
        "base_loss_weight": args.base_loss_weight if has_refiner else 0.0,
        "error_loss_weight": args.error_loss_weight if has_refiner else 0.0,
        "train_seed": args.train_seed,
        "epoch_seed_formula": "train_seed * 10000 + epoch",
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
    initial_phase = "head_warmup" if has_new_module else "tail"
    configure_trainable_parameters(model, model_name, initial_phase)
    optimizer = build_optimizer(
        model, model_name, args.head_learning_rate, args.tail_learning_rate
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, args.workers, args.train_seed + 900000, False
    )
    initial_metrics, _ = validation_metrics(
        model, validation_loader, criterion, device, returns_aux
    )
    best_metrics = dict(initial_metrics)
    best_epoch = 0
    best_score = initial_metrics["validation_soft_dice"]
    save_checkpoint(
        checkpoint_path, model, model_name, 0, initial_metrics, config, args.dataset
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
            "head_learning_rate": args.head_learning_rate if has_new_module else 0.0,
            "is_best": 1,
        }
    ]
    print(
        f"{model_name} epoch 0: loss={initial_metrics['validation_loss']:.6f}, "
        f"hard={initial_metrics['validation_dice']:.6f}, "
        f"soft={initial_metrics['validation_soft_dice']:.6f}"
    )

    for epoch in range(1, args.epochs + 1):
        phase = (
            "head_warmup"
            if has_new_module and epoch <= args.warmup_epochs
            else "tail"
        )
        configure_trainable_parameters(model, model_name, phase)
        model.train()
        epoch_seed = args.train_seed * 10000 + epoch
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % (2**32))
        torch.manual_seed(epoch_seed)
        train_loader = make_loader(
            train_dataset, args.batch_size, args.workers, epoch_seed, True
        )
        sums = {
            key: 0.0
            for key in ("total", "final", "base", "fn", "fp", "input_delta", "logit_delta")
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

            if returns_aux:
                logits, auxiliary = model(images, return_aux=True)
                final_loss = criterion(logits, masks)
                base_loss = criterion(auxiliary["base_logits"], masks)
                input_delta = auxiliary["input_delta"].abs().mean()
                logit_delta = auxiliary["logit_delta"].abs().mean()
                if has_refiner:
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
                else:
                    fn_loss = fp_loss = final_loss.detach().new_zeros(())
                    loss = final_loss
            else:
                logits = model(images)
                final_loss = criterion(logits, masks)
                base_loss = final_loss
                fn_loss = fp_loss = final_loss.detach().new_zeros(())
                input_delta = logit_delta = final_loss.detach().new_zeros(())
                loss = final_loss

            loss.backward()
            optimizer.step()
            count = images.size(0)
            for key, value in (
                ("total", loss),
                ("final", final_loss),
                ("base", base_loss),
                ("fn", fn_loss),
                ("fp", fp_loss),
                ("input_delta", input_delta),
                ("logit_delta", logit_delta),
            ):
                sums[key] += value.item() * count
            progress.set_postfix(loss=f"{loss.item():.4f}")

        validation_loader = make_loader(
            validation_dataset, args.batch_size, args.workers, args.train_seed + 900000, False
        )
        metrics, _ = validation_metrics(
            model, validation_loader, criterion, device, returns_aux
        )
        improved = metrics["validation_soft_dice"] > best_score
        if improved:
            best_score = metrics["validation_soft_dice"]
            best_epoch = epoch
            best_metrics = dict(metrics)
            save_checkpoint(
                checkpoint_path, model, model_name, epoch, metrics, config, args.dataset
            )
        sample_count = len(train_dataset)
        rows.append(
            {
                "epoch": epoch,
                "phase": phase,
                "train_total_loss": sums["total"] / sample_count,
                "train_final_loss": sums["final"] / sample_count,
                "train_base_loss": sums["base"] / sample_count,
                "train_fn_loss": sums["fn"] / sample_count,
                "train_fp_loss": sums["fp"] / sample_count,
                "mean_abs_input_delta": sums["input_delta"] / sample_count,
                "mean_abs_logit_delta": sums["logit_delta"] / sample_count,
                **metrics,
                "tail_learning_rate": args.tail_learning_rate,
                "head_learning_rate": args.head_learning_rate if has_new_module else 0.0,
                "is_best": int(improved),
            }
        )
        write_csv(log_path, rows)
        print(
            f"{model_name} epoch {epoch}: val_loss={metrics['validation_loss']:.6f}, "
            f"hard={metrics['validation_dice']:.6f}, "
            f"soft={metrics['validation_soft_dice']:.6f}, "
            f"best={best_score:.6f} (epoch {best_epoch})"
        )

    best_checkpoint = load_torch_checkpoint(checkpoint_path)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    validation_loader = make_loader(
        validation_dataset, args.batch_size, args.workers, args.train_seed + 900000, False
    )
    confirmed_metrics, per_image_rows = validation_metrics(
        model, validation_loader, criterion, device, returns_aux, include_rows=True
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
    if returns_aux:
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

    all_pairs = discover_pairs(args.data_root, args.dataset, "train")
    train_pairs, validation_pairs, split_record = load_explicit_split(
        all_pairs, args.split_file, args.dataset
    )
    checkpoint = load_torch_checkpoint(args.initial_checkpoint)
    checkpoint_hash = file_sha256(args.initial_checkpoint)

    variant_names = list(dict.fromkeys(args.variants))
    input_modes = sorted({INPUT_MODES[name] for name in variant_names} | {"gray_rgb"})
    train_datasets = {
        mode: RetinalDataset(
            train_pairs, args.image_size, input_mode=mode, augment=True
        )
        for mode in input_modes
    }
    validation_datasets = {
        mode: RetinalDataset(
            validation_pairs, args.image_size, input_mode=mode, augment=False
        )
        for mode in input_modes
    }
    models = {}
    for model_name in variant_names:
        set_seed(args.train_seed)
        models[model_name] = MODEL_BUILDERS[model_name]()
    missing = {}
    for model_name, model in models.items():
        set_seed(args.train_seed)
        missing[model_name] = load_gn_weights(
            model, checkpoint, model_name, args.dataset
        )

    set_seed(args.train_seed)
    control = GrayRGBControlUNet()
    load_gn_weights(
        control, checkpoint, "gray_rgb_control_30_unet", args.dataset
    )
    samples_by_mode = {}
    for mode, dataset in validation_datasets.items():
        loader = make_loader(dataset, 1, 0, args.train_seed + 700000, False)
        sample, _, _ = next(iter(loader))
        samples_by_mode[mode] = sample[:, :, :128, :128]
    differences = measure_initial_differences(control, models, samples_by_mode)
    del control

    print(f"Device: {device}")
    print(f"Explicit split: train={len(train_pairs)}, validation={len(validation_pairs)}")
    print(f"Initial checkpoint SHA256: {checkpoint_hash}")
    for model_name in variant_names:
        print(
            f"{model_name}: new_tensors={len(missing[model_name])}, "
            f"initial_max_abs_difference={differences.get(model_name, 0.0):.9g}"
        )
    print(
        f"Official {args.dataset} test set is reserved and was not discovered or accessed."
    )

    results = {}
    for model_name in variant_names:
        model = models.pop(model_name).to(device)
        input_mode = INPUT_MODES[model_name]
        results[model_name] = train_variant(
            model,
            model_name,
            train_datasets[input_mode],
            validation_datasets[input_mode],
            split_record,
            args,
            device,
            checkpoint_hash,
            differences.get(model_name, 0.0),
        )
        model.cpu()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Ablation training complete:")
    for model_name in variant_names:
        print(
            f"  {model_name}: soft={results[model_name]['best_validation_soft_dice']:.6f}, "
            f"hard={results[model_name]['best_validation_dice']:.6f}"
        )


if __name__ == "__main__":
    main()
