"""Fixed three-arm CARER component audit. The training phase never opens test data."""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "frozen_base"))

import numpy as np
import PIL
import scipy
import torch
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

from component_models import ARMS, build_model, load_gn_weights
from data_utils import RetinalDataset, discover_pairs
from evaluation import calculate_metrics
from models.Attention_UNet_V5 import GrayRGBControlUNet
from structure_metrics import skeletonize_zhang_suen
from train_carer_paired import (
    BCEDiceLoss, file_sha256, load_explicit_split, load_torch_checkpoint,
    make_loader, resolve_device, set_seed, validation_metrics,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def protocol():
    path = PACKAGE / "protocol.json"
    record = read_json(path)
    if record["arms"] != list(ARMS) or record["seeds"] != [42, 123, 3407]:
        raise ValueError("Unexpected protocol arms or seeds")
    return record, file_sha256(path)


def check_package():
    path = PACKAGE / "package_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("package_manifest.json is missing; use the completed distribution package")
    manifest = read_json(path)
    for row in manifest["files"]:
        relative = Path(row["path"])
        actual = (PACKAGE / relative).resolve()
        if not actual.is_relative_to(PACKAGE.resolve()):
            raise ValueError("Manifest path leaves the package")
        if not actual.is_file() or file_sha256(actual) != row["sha256"]:
            raise ValueError(f"Package integrity mismatch: {relative}")
    return file_sha256(path)


def environment(device):
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pillow": PIL.__version__, "scipy": scipy.__version__,
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def source_checkpoint(project_root, seed):
    return Path(project_root) / "weight_files_fives_v2_screen" / f"FIVES_best_ResAttentionUNetGN_seed{seed}.pth"


def split_path(seed):
    return PACKAGE / "reference_splits" / f"seed_{seed}.json"


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_finite(metrics):
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise FloatingPointError(f"Non-finite value in {key}: {value}")


def dataset_manifest(pairs, role="training pool only"):
    records = []
    for image, mask, key in tqdm(pairs, desc=f"Hashing {len(pairs)} {role} pairs", ncols=100):
        entry = {"sample": key}
        for role, path in (("image", image), ("mask", mask)):
            stat = path.stat()
            with Image.open(path) as original:
                canonical=original.convert("RGB" if role=="image" else "L").resize(
                    (512,512),Image.Resampling.BILINEAR if role=="image" else Image.Resampling.NEAREST)
                pixels=np.asarray(canonical,dtype=np.uint8)
                if role=="mask":pixels=(pixels>127).astype(np.uint8)
                pixel_hash=hashlib.sha256(pixels.tobytes()).hexdigest()
            entry[role] = {"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "sha256": file_sha256(path),"pixel_sha256":pixel_hash}
        records.append(entry)
    digest = object_hash([{ "sample": row["sample"], "image": row["image"]["sha256"], "mask": row["mask"]["sha256"]} for row in records])
    return {"role": role, "samples": len(records), "data_sha256": digest, "files": records}


def verify_dataset_metadata(pairs, manifest):
    by_key = {row["sample"]: row for row in manifest["files"]}
    if set(by_key) != {key for _, _, key in pairs}:
        raise ValueError("Training data keys changed since preflight")
    for image, mask, key in pairs:
        for role, path in (("image", image), ("mask", mask)):
            stat = path.stat()
            old = by_key[key][role]
            if stat.st_size != old["bytes"] or stat.st_mtime_ns != old["mtime_ns"]:
                raise ValueError(f"Training data metadata changed since preflight: {key}/{role}; run preflight into a new output directory")


def verify_disjoint_image_contents(training_record, test_record):
    """FIVES basenames are split-local; leakage checks must use image content."""
    trained_bytes={row["image"]["sha256"] for row in training_record["files"]}
    trained_pixels={row["image"]["pixel_sha256"] for row in training_record["files"]}
    repeated=[row["sample"] for row in test_record["files"]
              if row["image"]["sha256"] in trained_bytes or row["image"]["pixel_sha256"] in trained_pixels]
    if repeated:
        raise ValueError(f"Actual test image content duplicates training-pool images: {repeated[:8]}; names alone are not used for this check")


def get_train_pairs(data_root):
    pairs = discover_pairs(data_root, "FIVES", "train")
    if len(pairs) != 600:
        raise ValueError(f"Expected exactly 600 FIVES training pairs, found {len(pairs)}")
    return pairs


def validate_checkpoint(project_root, seed, spec):
    path = source_checkpoint(project_root, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Historical GN checkpoint missing: {path}")
    actual = file_sha256(path)
    if actual.lower() != spec["expected_gn_checkpoint_sha256"][str(seed)].lower():
        raise ValueError(f"Seed {seed}: source checkpoint differs from the historical GN reference; no substitution or retraining is permitted")
    return path, actual


def preflight(args):
    spec, phash = protocol()
    package_hash = check_package()
    if not hasattr(np,"trapezoid"):
        raise RuntimeError("The frozen evaluation code requires NumPy >= 2.0; see requirements_component.txt. No environment packages were changed automatically.")
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The full experiment requires the 4070 CUDA environment; synthetic unit tests may use CPU")
    if torch.cuda.get_device_properties(device).total_memory < 10 * 1024**3:
        raise RuntimeError("This FP32 package requires at least 10 GiB GPU memory; do not silently change precision or image size")
    set_seed(42)
    pairs = get_train_pairs(args.data_root)
    records = dataset_manifest(pairs)
    previous = args.output_root / "preflight.json"
    if previous.exists():
        old = read_json(previous)
        if old["protocol_hash"] != phash or old["package_manifest_sha256"] != package_hash or old["data_sha256"] != records["data_sha256"]:
            raise ValueError("Output directory belongs to different code, protocol, or data; choose a new output directory")
        if old["environment"] != environment(device):
            raise ValueError("Execution environment changed; choose a new output directory rather than mixing runs")
    sample = None
    checkpoints = {}
    for seed in spec["seeds"]:
        train_pairs, val_pairs, _ = load_explicit_split(pairs, split_path(seed), "FIVES")
        if len(train_pairs) != 540 or len(val_pairs) != 60:
            raise ValueError("Historical split is not 540/60")
        path, digest = validate_checkpoint(args.project_root, seed, spec)
        checkpoints[str(seed)] = {"sha256": digest, "split_sha256": file_sha256(split_path(seed))}
        if seed != 42:
            continue
        images, _, _ = RetinalDataset(val_pairs[:1], 512, "gray_rgb", False)[0]
        if torch.equal(images[1], images[2]) and torch.equal(images[2], images[3]):
            raise ValueError("FIVES sample appears grayscale-replicated; native RGB files are required")
        sample = images.unsqueeze(0).to(device)
        checkpoint = load_torch_checkpoint(path)
        set_seed(seed)
        gray = GrayRGBControlUNet().to(device).eval()
        gray.load_state_dict(checkpoint["state_dict"], strict=True)
        with torch.inference_mode():
            reference = gray(sample).cpu()
        del gray
        for arm in ARMS:
            set_seed(seed)
            model = build_model(arm)
            load_gn_weights(model, checkpoint)
            if sum(p.numel() for p in model.parameters()) != 32779990:
                raise AssertionError("Unexpected model parameter count")
            model.to(device).eval()
            with torch.inference_mode():
                logits, aux = model(sample, return_aux=True)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Invalid initial output: {arm}")
            maximum = float((logits.cpu() - reference).abs().max())
            if arm in ("core", "no_bound") and maximum > 1e-6:
                raise AssertionError(f"Initialization equivalence failed: {arm}")
            print(f"512px GPU preflight {arm}: max initial logit difference={maximum:.9g}; mean input correction={float(aux['input_delta'].abs().mean()):.9g}", flush=True)
            del model, logits, aux
            torch.cuda.empty_cache()
        del checkpoint, sample
        gc.collect()
    payload = {"passed": True, "protocol_hash": phash, "package_manifest_sha256": package_hash,
               "data_sha256": records["data_sha256"], "workers": args.workers,
               "environment": environment(device), "checkpoints": checkpoints,
               "official_test_accessed_by_this_phase": False,
               "historical_test_exposure": True}
    write_json(args.output_root / "training_data_manifest.json", records)
    write_json(previous, payload)
    print("PREFLIGHT PASSED. Only the 600-image training pool was opened.", flush=True)


def configure_parameters(model, warmup):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.color_detail_stem.parameters():
        parameter.requires_grad = True
    if not warmup:
        for module in (model.d1, model.out_conv):
            for parameter in module.parameters():
                parameter.requires_grad = True


def optimizer_for(model, recipe):
    return torch.optim.Adam([
        {"params": list(model.d1.parameters()) + list(model.out_conv.parameters()), "lr": recipe["tail_learning_rate"], "name": "gn_tail"},
        {"params": list(model.color_detail_stem.parameters()), "lr": recipe["module_learning_rate"], "name": "new_module"},
    ], betas=tuple(recipe["optimizer_betas"]), eps=recipe["optimizer_epsilon"], weight_decay=recipe["weight_decay"])


def initial_diagnostics(model, sample):
    model.eval()
    with torch.inference_mode():
        logits, aux = model(sample, return_aux=True)
        delta = aux["input_delta"]
        raw = aux["raw_residual"]
        # Bypass the adapter to measure the identical backbone's grayscale start.
        from models.Attention_UNet_V2 import ImprovedAttentionUNetV2
        gray_logits = ImprovedAttentionUNetV2.forward(model, sample[:, :1])
        return {"initial_max_abs_logit_difference": float((logits-gray_logits).abs().max()),
                "initial_mean_abs_input_delta": float(delta.abs().mean()),
                "initial_max_abs_input_delta": float(delta.abs().max()),
                "initial_raw_residual_std": float(raw.std(unbiased=False)),
                "initial_adjusted_input_outside_0_1_fraction": float(((aux["adjusted_input"]<0) | (aux["adjusted_input"]>1)).float().mean())}


def hash_batch(digest, images, masks, keys):
    for key in keys:
        digest.update(key.encode("utf-8") + b"\0")
    digest.update(images.contiguous().numpy().tobytes())
    digest.update(masks.contiguous().numpy().tobytes())


def train_engine(model, train_data, val_data, config, run_dir, checkpoint_dir, device, *, test_only=False):
    """Shared engine; test_only is only used by synthetic unit tests, never by CLI."""
    recipe = config["recipe"]
    run_dir, checkpoint_dir = Path(run_dir), Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing run config differs; refusing to mix results")
    if not config_path.exists():
        write_json(config_path, config)
    config_hash = file_sha256(config_path)
    result_path = run_dir / "best_result.json"
    best_path, last_path = checkpoint_dir / "best.pth", checkpoint_dir / "last.pth"
    if result_path.exists():
        result = read_json(result_path)
        if not result.get("completed") or result["config_sha256"] != config_hash or not best_path.is_file() or result["best_checkpoint_sha256"] != file_sha256(best_path):
            raise ValueError("Completed result has an integrity mismatch")
        print(f"SKIP verified completed run {config['arm']}/seed_{config['seed']}", flush=True)
        return result
    criterion = BCEDiceLoss()
    model.to(device)
    configure_parameters(model, True)
    optimizer = optimizer_for(model, recipe)
    seed = config["seed"]
    if last_path.exists():
        last = load_torch_checkpoint(last_path)
        if last["config_sha256"] != config_hash:
            raise ValueError("Resume checkpoint config hash mismatch")
        model.load_state_dict(last["state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        rows = last["rows"]
        best_state, best_metrics = last["best_state_dict"], last["best_metrics"]
        best_epoch = last["best_epoch"]
        first_epoch = int(last["completed_epoch"]) + 1
        if [row["epoch"] for row in rows] != list(range(first_epoch)):
            raise ValueError("Resume training log is not contiguous")
        del last
        print(f"RESUME {config['arm']}/seed_{seed} at epoch {first_epoch}", flush=True)
    else:
        loader = make_loader(val_data, 1, config["workers"], seed+900000, False)
        metrics, _ = validation_metrics(model, loader, criterion, device, True)
        verify_finite(metrics)
        sample = val_data[0][0].unsqueeze(0).to(device)
        diagnostic = initial_diagnostics(model, sample)
        del sample
        best_state, best_metrics, best_epoch = cpu_state(model), dict(metrics), 0
        rows = [{"epoch": 0, "phase": "initial", "train_total_loss": "", "mean_abs_input_delta": diagnostic["initial_mean_abs_input_delta"],
                 "max_abs_input_delta": diagnostic["initial_max_abs_input_delta"], "fraction_abs_input_delta_gt_0_1": 0.0,
                 "adjusted_input_outside_0_1_fraction": diagnostic["initial_adjusted_input_outside_0_1_fraction"],
                 "raw_residual_abs_ge_3_fraction": "", "augmentation_stream_sha256": "", "train_samples": 0,
                 "module_learning_rate": recipe["module_learning_rate"], "tail_learning_rate": recipe["tail_learning_rate"],
                 "elapsed_seconds": 0.0, "is_best": 1, **metrics}]
        write_json(run_dir / "initial_diagnostics.json", diagnostic)
        first_epoch = 1
    best_score = best_metrics["validation_soft_dice"]
    for epoch in range(first_epoch, recipe["epochs"]+1):
        epoch_start = time.perf_counter()
        warmup = epoch <= recipe["warmup_epochs"]
        configure_parameters(model, warmup)
        model.train()
        epoch_seed = seed*10000+epoch
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % 2**32)
        torch.manual_seed(epoch_seed)
        loader = make_loader(train_data, 1, config["workers"], epoch_seed, True)
        stream = hashlib.sha256()
        sums = dict(loss=0.0, abs_delta=0.0, outside=0.0, above_bound=0.0, raw_saturated=0.0)
        maximum, seen = 0.0, 0
        for images, masks, keys in tqdm(loader, desc=f"{config['arm']} seed {seed} epoch {epoch}/{recipe['epochs']}", ncols=110):
            hash_batch(stream, images, masks, keys)
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(images, return_aux=True)
            loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: arm={config['arm']} seed={seed} epoch={epoch}; no automatic LR or bound change")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Non-finite gradient in {name}")
            optimizer.step()
            count = images.shape[0]
            delta = aux["input_delta"].detach()
            adjusted = aux["adjusted_input"].detach()
            raw = aux["raw_residual"].detach()
            sums["loss"] += float(loss.detach())*count
            sums["abs_delta"] += float(delta.abs().mean())*count
            sums["outside"] += float(((adjusted<0) | (adjusted>1)).float().mean())*count
            sums["above_bound"] += float((delta.abs()>0.100001).float().mean())*count
            sums["raw_saturated"] += float((raw.abs()>=3).float().mean())*count
            maximum = max(maximum, float(delta.abs().max()))
            seen += count
        if seen != len(train_data):
            raise RuntimeError("Training loader did not visit every training image once")
        if config["arm"] != "no_bound" and maximum > 0.100001:
            raise AssertionError("Bounded arm exceeded the specified residual bound")
        val_loader = make_loader(val_data, 1, config["workers"], seed+900000, False)
        metrics, _ = validation_metrics(model, val_loader, criterion, device, True)
        verify_finite(metrics)
        improved = metrics["validation_soft_dice"] > best_score
        if improved:
            best_state, best_metrics, best_epoch = cpu_state(model), dict(metrics), epoch
            best_score = metrics["validation_soft_dice"]
        row = {"epoch": epoch, "phase": "head_warmup" if warmup else "tail", "train_total_loss": sums["loss"]/seen,
               "mean_abs_input_delta": sums["abs_delta"]/seen, "max_abs_input_delta": maximum,
               "fraction_abs_input_delta_gt_0_1": sums["above_bound"]/seen,
               "adjusted_input_outside_0_1_fraction": sums["outside"]/seen,
               "raw_residual_abs_ge_3_fraction": sums["raw_saturated"]/seen,
               "augmentation_stream_sha256": stream.hexdigest(), "train_samples": seen,
               "module_learning_rate": recipe["module_learning_rate"], "tail_learning_rate": recipe["tail_learning_rate"],
               "elapsed_seconds": time.perf_counter()-epoch_start, "is_best": int(improved), **metrics}
        rows.append(row)
        write_csv(run_dir / "training_log.csv", rows)
        # One atomic resume transaction also contains its own best weights.
        save_torch(last_path, {"state_dict": cpu_state(model), "optimizer": optimizer.state_dict(),
                              "best_state_dict": best_state, "best_metrics": best_metrics,
                              "best_epoch": best_epoch, "completed_epoch": epoch,
                              "rows": rows, "config_sha256": config_hash})
        print(f"epoch {epoch}: validation soft Dice={metrics['validation_soft_dice']:.7f}; best epoch={best_epoch}", flush=True)
    model.load_state_dict(best_state, strict=True)
    if "frozen_backbone_state_sha256" in config:
        current_frozen=state_hash({name:value for name,value in model.state_dict().items()
                                  if not name.startswith(("color_detail_stem.","d1.","out_conv."))})
        if current_frozen!=config["frozen_backbone_state_sha256"]:
            raise AssertionError("Frozen backbone tensors changed during adaptation")
    val_loader = make_loader(val_data, 1, config["workers"], seed+900000, False)
    confirmed, per_image = validation_metrics(model, val_loader, criterion, device, True, include_rows=True)
    verify_finite(confirmed)
    if abs(confirmed["validation_soft_dice"]-best_metrics["validation_soft_dice"]) > 1e-6:
        raise RuntimeError("Reloaded best checkpoint does not reproduce its validation selection value")
    write_csv(run_dir / "validation_per_image.csv", per_image)
    save_torch(best_path, {"state_dict": best_state, "arm": config["arm"], "seed": seed, "dataset": "FIVES",
                           "epoch": best_epoch, "config": config, "config_sha256": config_hash,
                           "protocol_hash": config["protocol_hash"], "selection_metric": "soft_dice", "metrics": confirmed})
    result = {"completed": True, "completed_epochs": recipe["epochs"], "best_epoch": best_epoch,
              "selection_metric": "soft_dice", "best_validation_loss": confirmed["validation_loss"],
              "best_validation_dice": confirmed["validation_dice"], "best_validation_soft_dice": confirmed["validation_soft_dice"],
              "best_checkpoint": str(best_path.relative_to(config["output_root"])) if not test_only else str(best_path),
              "best_checkpoint_sha256": file_sha256(best_path), "config_sha256": config_hash,
              "protocol_hash": config["protocol_hash"], "split_sha256": config["split_sha256"],
              "initial_checkpoint_sha256": config["initial_checkpoint_sha256"], "official_test_accessed": False,
              "historical_test_exposure": True, "synthetic_test_only": test_only}
    write_json(result_path, result)
    return result


def train(args):
    spec, phash = protocol()
    package_hash = check_package()
    if args.seed not in spec["seeds"]:
        raise ValueError("Seed is not in the fixed three-seed plan")
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Full training must use CUDA, not an unintended CPU fallback")
    set_seed(args.seed)
    pre = read_json(args.output_root / "preflight.json")
    if not pre["passed"] or pre["protocol_hash"] != phash or pre["package_manifest_sha256"] != package_hash or pre["environment"] != environment(device) or pre["workers"] != args.workers:
        raise ValueError("Preflight contract differs from this training invocation")
    path, checkpoint_hash = validate_checkpoint(args.project_root, args.seed, spec)
    pairs = get_train_pairs(args.data_root)
    verify_dataset_metadata(pairs, read_json(args.output_root / "training_data_manifest.json"))
    train_pairs, val_pairs, split = load_explicit_split(pairs, split_path(args.seed), "FIVES")
    set_seed(args.seed)
    model = build_model(args.arm)
    load_gn_weights(model, load_torch_checkpoint(path))
    recipe = spec["shared"]
    initial_state=model.state_dict()
    output_prefix="color_detail_stem.input_delta."
    frozen_prefixes=("color_detail_stem.","d1.","out_conv.")
    config = {"arm": args.arm, "seed": args.seed, "train_seed": args.seed, "dataset": "FIVES",
              "protocol_hash": phash, "package_manifest_sha256": package_hash, "recipe": recipe,
              "epochs": recipe["epochs"], "warmup_epochs": recipe["warmup_epochs"], "image_size": 512,
              "batch_size": 1, "head_learning_rate": recipe["module_learning_rate"], "tail_learning_rate": recipe["tail_learning_rate"],
              "input_mode": "gray_rgb", "selection_metric": "soft_dice", "workers": args.workers,
              "parameter_count": 32779990, "initial_checkpoint_sha256": checkpoint_hash,
              "split_sha256": file_sha256(split_path(args.seed)), "data_sha256": pre["data_sha256"],
              "environment": environment(device), "output_root": str(args.output_root.resolve()),
              "initial_shared_state_sha256":state_hash({k:v for k,v in initial_state.items() if not k.startswith(output_prefix)}),
              "initial_full_state_sha256":state_hash(initial_state),
              "initial_output_head_sha256":state_hash({k:v for k,v in initial_state.items() if k.startswith(output_prefix)}),
              "frozen_backbone_state_sha256":state_hash({k:v for k,v in initial_state.items() if not k.startswith(frozen_prefixes)}),
              "study_role": spec["study_role"]}
    del initial_state
    run_dir = args.output_root / "runs" / args.arm / f"seed_{args.seed}"
    checkpoint_dir = args.output_root / "checkpoints" / args.arm / f"seed_{args.seed}"
    write_json(run_dir / "split.json", split)
    train_data = RetinalDataset(train_pairs, 512, "gray_rgb", True)
    val_data = RetinalDataset(val_pairs, 512, "gray_rgb", False)
    try:
        train_engine(model, train_data, val_data, config, run_dir, checkpoint_dir, device)
    except Exception as exc:
        write_json(run_dir / "failure.json", {"arm": args.arm, "seed": args.seed, "exception": type(exc).__name__,
                    "message": str(exc), "protocol_hash": phash, "automatic_recipe_change": False})
        raise


def seal(args):
    spec, phash = protocol()
    check_package()
    records = []
    for seed in spec["seeds"]:
        streams = []
        configs=[]
        for arm in ARMS:
            run_dir = args.output_root / "runs" / arm / f"seed_{seed}"
            cfg, res = read_json(run_dir/"config.json"), read_json(run_dir/"best_result.json")
            configs.append(cfg)
            best = args.output_root / res["best_checkpoint"]
            if not res["completed"] or res["completed_epochs"] != 30 or cfg["protocol_hash"] != phash or res["protocol_hash"] != phash:
                raise ValueError(f"Incomplete fixed training: {arm}/{seed}")
            if file_sha256(best) != res["best_checkpoint_sha256"] or file_sha256(run_dir/"config.json") != res["config_sha256"]:
                raise ValueError("Training artifact integrity mismatch")
            with (run_dir/"training_log.csv").open(encoding="utf-8-sig",newline="") as handle:
                rows=list(csv.DictReader(handle))
            if [int(row["epoch"]) for row in rows] != list(range(31)):
                raise ValueError("Training log does not contain all epochs 0 through 30")
            streams.append([row["augmentation_stream_sha256"] for row in rows[1:]])
            records.append({"arm": arm, "seed": seed, "config_sha256": file_sha256(run_dir/"config.json"),
                            "best_result_sha256": file_sha256(run_dir/"best_result.json"),
                            "best_checkpoint_sha256": file_sha256(best), "best_checkpoint": res["best_checkpoint"],
                            "initial_checkpoint_sha256": cfg["initial_checkpoint_sha256"], "split_sha256": cfg["split_sha256"]})
        if streams[0] != streams[1] or streams[0] != streams[2]:
            raise ValueError(f"Seed {seed}: training sample/augmentation tensors differ across arms; paired audit cannot proceed")
        if len({cfg["initial_shared_state_sha256"] for cfg in configs})!=1:
            raise ValueError(f"Seed {seed}: non-output-layer initial tensors differ across arms")
        if configs[0]["initial_full_state_sha256"]!=configs[1]["initial_full_state_sha256"]:
            raise ValueError(f"Seed {seed}: core and no_bound did not start from identical tensors")
    payload={"complete": True, "protocol_hash": phash, "runs": records,
             "all_augmentation_streams_identical_within_seed": True,
             "new_test_inference_started": False, "historical_test_exposure": True}
    path=args.output_root/"training_seal.json"
    if path.exists() and read_json(path) != payload:
        raise ValueError("An existing seal differs; refusing to reseal different checkpoints")
    write_json(path,payload)
    print("SEALED: all nine validation-selected checkpoints, before new test inference.",flush=True)


def verify_seal(output_root):
    spec, phash=protocol()
    seal_record=read_json(output_root/"training_seal.json")
    if not seal_record.get("complete") or seal_record["protocol_hash"] != phash:
        raise ValueError("Missing or invalid all-nine training seal")
    wanted={(arm,seed) for arm in ARMS for seed in spec["seeds"]}
    if {(row["arm"],row["seed"]) for row in seal_record["runs"]} != wanted or len(seal_record["runs"]) != 9:
        raise ValueError("Seal does not contain exactly all nine planned runs")
    for row in seal_record["runs"]:
        run_dir=output_root/"runs"/row["arm"]/f"seed_{row['seed']}"
        for path,expected in [(run_dir/"config.json",row["config_sha256"]),(run_dir/"best_result.json",row["best_result_sha256"]),(output_root/row["best_checkpoint"],row["best_checkpoint_sha256"])]:
            if file_sha256(path) != expected:
                raise ValueError(f"Sealed artifact changed: {path}")
    return seal_record


def test(args):
    check_package()
    spec,phash=protocol()
    sealed=verify_seal(args.output_root)
    matching=[row for row in sealed["runs"] if row["arm"]==args.arm and row["seed"]==args.seed]
    if len(matching)!=1:
        raise ValueError("Requested test run is not in the seal")
    record=matching[0]
    run_dir=args.output_root/"runs"/args.arm/f"seed_{args.seed}"
    cfg=read_json(run_dir/"config.json")
    out=args.output_root/"results"/args.arm/f"seed_{args.seed}"
    finished=out/"metrics.json"
    device=resolve_device(args.device)
    set_seed(args.seed)
    if cfg["environment"] != environment(device):
        raise ValueError("Test environment differs from training environment")
    # Test files are deliberately first discovered only after the complete seal check.
    pairs=discover_pairs(args.data_root,"FIVES","test")
    if len(pairs)!=200:
        raise ValueError("Expected exactly 200 official FIVES test pairs")
    manifest_path=args.output_root/"official_test_data_manifest.json"
    data_record=dataset_manifest(pairs,"official test")
    training_record=read_json(args.output_root/"training_data_manifest.json")
    # Official train/test filenames can repeat (e.g. 11_A); compare contents, not basenames.
    verify_disjoint_image_contents(training_record,data_record)
    if manifest_path.exists():
        recorded=read_json(manifest_path)
        if recorded["data_sha256"]!=data_record["data_sha256"] or recorded["samples"]!=200:
            raise ValueError("Official-test image/mask contents differ from the first sealed-checkpoint evaluation")
    else:
        write_json(manifest_path,data_record)
    if finished.exists():
        old=read_json(finished)
        if old["protocol_hash"]!=phash or old["best_checkpoint_sha256"]!=record["best_checkpoint_sha256"] or old["per_image_csv_sha256"]!=file_sha256(out/"per_image_metrics.csv") or old.get("test_data_sha256")!=data_record["data_sha256"] or old.get("test_data_manifest_sha256")!=file_sha256(manifest_path):
            raise ValueError("Existing test result no longer matches its sealed artifacts/data")
        print(f"SKIP verified test result {args.arm}/seed_{args.seed}",flush=True)
        return
    checkpoint=load_torch_checkpoint(args.output_root/record["best_checkpoint"])
    if checkpoint["config_sha256"]!=record["config_sha256"]:
        raise ValueError("Selected checkpoint configuration mismatch")
    model=build_model(args.arm).to(device)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    model.eval()
    data=RetinalDataset(pairs,512,"gray_rgb",False)
    loader=make_loader(data,1,args.workers,args.seed+1900000,False)
    prediction_dir=out/"predictions"
    prediction_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.inference_mode():
        for images,masks,keys in tqdm(loader,desc=f"Fixed test {args.arm}/{args.seed}",ncols=100):
            logits=model(images.to(device,non_blocking=True))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite official-test logits")
            probability=torch.sigmoid(logits).cpu().numpy()[0,0]
            truth=masks.numpy()[0,0]>0.5
            fov=images.numpy()[0,0]>(5.0/255.0)
            prediction=probability>0.5
            metric=calculate_metrics(truth,probability,threshold=0.5,fov_mask=fov)
            truth_fov=truth&fov
            predicted_fov=prediction&fov
            skeleton=skeletonize_zhang_suen(truth_fov)
            radius=ndi.distance_transform_edt(truth_fov)
            thin=skeleton&(radius<=1.5)
            predicted_skeleton=skeletonize_zhang_suen(predicted_fov)
            eps=1e-8
            tp=float((predicted_skeleton&truth_fov).sum())/(float(predicted_skeleton.sum())+eps)
            tr=float((skeleton&predicted_fov).sum())/(float(skeleton.sum())+eps)
            metric["cldice"]=2*tp*tr/(tp+tr+eps)
            metric["thin_centerline_recall"]=float((thin&predicted_fov).sum())/(float(thin.sum())+eps)
            metric["thin_centerline_pixels"]=int(thin.sum())
            metric["thin_centerline_empty"]=int(thin.sum()==0)
            # Historical metric code defines zero for an empty thin reference; flag it explicitly.
            verify_finite(metric)
            row={"sample":keys[0],**metric}
            rows.append(row)
            Image.fromarray((prediction.astype(np.uint8)*255)).save(prediction_dir/f"{keys[0]}.png")
    write_csv(out/"per_image_metrics.csv",rows)
    numeric=[key for key in rows[0] if key!="sample"]
    summary={key:{"mean":float(np.mean([row[key] for row in rows])),"std":float(np.std([row[key] for row in rows],ddof=1))} for key in numeric}
    write_json(finished,{"arm":args.arm,"seed":args.seed,"test_samples":200,"threshold":0.5,
                "fov":"original grayscale > 5/255", "image_size":512,"protocol_hash":phash,
                "config_sha256":record["config_sha256"],"initial_checkpoint_sha256":cfg["initial_checkpoint_sha256"],
                "split_sha256":cfg["split_sha256"],"best_checkpoint_sha256":record["best_checkpoint_sha256"],
                "training_seal_sha256":file_sha256(args.output_root/"training_seal.json"),
                "test_data_sha256":data_record["data_sha256"],"test_data_manifest_sha256":file_sha256(manifest_path),
                "per_image_csv_sha256":file_sha256(out/"per_image_metrics.csv"),
                "historical_test_exposure":True,"test_based_selection":False,"metrics":summary})
    print(f"Fixed official test completed: {args.arm}/seed_{args.seed}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["preflight","train","seal","test","check-package"])
    parser.add_argument("--project-root",type=Path,default=PACKAGE.parent)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--output-root",type=Path,default=PACKAGE/"outputs")
    parser.add_argument("--arm",choices=ARMS)
    parser.add_argument("--seed",type=int,choices=[42,123,3407])
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    args=parser.parse_args()
    if args.workers<0 or args.workers>16:
        raise ValueError("workers must be between 0 and 16")
    args.project_root=args.project_root.resolve()
    args.output_root=args.output_root.resolve()
    args.data_root=(args.data_root or args.project_root/"data").resolve()
    if args.stage in ("train","test") and (args.arm is None or args.seed is None):
        parser.error("train/test require --arm and --seed")
    if args.stage=="check-package":
        print(check_package())
    else:
        {"preflight":preflight,"train":train,"seal":seal,"test":test}[args.stage](args)


if __name__=="__main__":
    main()
