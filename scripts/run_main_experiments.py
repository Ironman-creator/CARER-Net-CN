"""中文主实验计划器。默认只展示命令；显式 --execute 才会训练或测试。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 3407)
ALIASES = {"gn": "res_attention_unet_gn", "baseline": "gray_rgb_control_30_unet",
           "core": "color_detail_only_unet", "refine": "carer_unet"}
TAGS = {"res_attention_unet_gn": "ResAttentionUNetGN", "unet": "UNet",
        "attention_unet": "AttentionUNet", "unet_plus_plus_gn": "UNetPlusPlusGN",
        "r2_unet_gn": "R2UNetGN", "sa_unet_gn": "SAUNetGN"}
COMPARATORS = ("unet", "attention_unet", "unet_plus_plus_gn", "r2_unet_gn", "sa_unet_gn")
ADAPTERS = ("gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet")


def build_plan(args):
    data = args.data_root.resolve()
    output = args.output_root.resolve()
    runs, weights, results = (output / name for name in ("runs", "weights", "results"))
    requested = list(dict.fromkeys(ALIASES.get(name, name) for name in args.models))
    if args.all_baselines:
        requested.extend(name for name in COMPARATORS if name not in requested)
    adapt = [name for name in ADAPTERS if name in requested]
    if adapt and ADAPTERS[0] not in adapt:
        adapt.insert(0, ADAPTERS[0])
    base = [name for name in requested if name in TAGS]
    if adapt and "res_attention_unet_gn" not in base:
        base.insert(0, "res_attention_unet_gn")
    evaluate = list(dict.fromkeys(base + adapt))
    training, testing = [], []
    for dataset in dict.fromkeys(args.datasets):
        epochs, minimum, patience = (100, 80, 25) if dataset == "STARE" else (50, 30, 15)
        for seed in dict.fromkeys(args.seeds):
            split = ROOT / f"splits/main/{dataset}/seed_{seed}.json"
            if not split.is_file():
                raise FileNotFoundError(split)
            for model in base:
                command = [sys.executable, str(ROOT / "train.py"), "--dataset", dataset,
                    "--model", model, "--data-root", str(data), "--split-file", str(split),
                    "--validation-ratio", "0.1" if dataset == "FIVES" else "0.2",
                    "--run-root", str(runs), "--weight-dir", str(weights),
                    "--image-size", "512", "--input-mode", "grayscale", "--epochs", str(epochs),
                    "--min-epochs", str(minimum), "--patience", str(patience), "--batch-size", "1",
                    "--learning-rate", "0.0001", "--loss", "bce_dice", "--bce-weight", "1",
                    "--dice-weight", "1", "--selection-metric", "soft_dice", "--seed", str(seed),
                    "--workers", str(args.workers), "--device", args.device, "--deterministic"]
                training.append({"dataset": dataset, "seed": seed, "model": model, "command": command})
            if adapt:
                initial = weights / f"{dataset}_best_ResAttentionUNetGN_seed{seed}.pth"
                command = [sys.executable, str(ROOT / "train_carer_ablation.py"), "--dataset", dataset,
                    "--data-root", str(data), "--split-file", str(split), "--initial-checkpoint", str(initial),
                    "--run-root", str(runs), "--weight-dir", str(weights), "--image-size", "512",
                    "--batch-size", "1", "--epochs", "30", "--warmup-epochs", "5",
                    "--head-learning-rate", "0.0001", "--tail-learning-rate", "0.00001",
                    "--base-loss-weight", "0.25", "--error-loss-weight", "0.10", "--train-seed", str(seed),
                    "--workers", str(args.workers), "--device", args.device, "--variants", *adapt]
                training.append({"dataset": dataset, "seed": seed, "model": ";".join(adapt), "command": command})
            for model in evaluate:
                filename = f"{dataset}_best_{model if model in ADAPTERS else TAGS[model]}_seed{seed}.pth"
                command = [sys.executable, str(ROOT / "test.py"), "--dataset", dataset, "--model", model,
                    "--checkpoint", str(weights / filename), "--data-root", str(data),
                    "--result-root", str(results), "--run-root", str(runs), "--image-size", "512",
                    "--input-mode", "gray_rgb" if model in ADAPTERS else "grayscale",
                    "--threshold", "0.5", "--fov", "derived", "--seed", str(seed),
                    "--workers", str(args.workers), "--device", args.device]
                testing.append({"dataset": dataset, "seed": seed, "model": model, "command": command})
    return {"training": training, "testing": testing,
            "output_root": str(output), "execution_requested": args.execute,
            "note": "所有计划训练结束后才开始计划测试；不覆盖已有训练。"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("FIVES", "DRIVE", "STARE", "CHASE"), default=["FIVES"])
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--models", nargs="+", choices=tuple(ALIASES) + COMPARATORS, default=list(ALIASES))
    parser.add_argument("--all-baselines", action="store_true", help="加入表2的五个比较模型")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/main_reproduction")
    parser.add_argument("--workers", type=int, choices=range(0, 17), default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--phase", choices=("all", "train", "test"), default="all")
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    plan = build_plan(args)
    if args.plan_out:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = ([] if args.phase == "test" else plan["training"]) + ([] if args.phase == "train" else plan["testing"])
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print(f"训练命令 {len(plan['training'])} 条；测试命令 {len(plan['testing'])} 条。")
        for item in selected:
            print(subprocess.list2cmdline(item["command"]) if sys.platform == "win32" else shlex.join(item["command"]))
        if not args.execute:
            print("计划模式：未训练、未测试。确认数据与路径后加 --execute。")
    if args.execute:
        for dataset in args.datasets:
            for directory in ("train_images", "train_masks", "test_images", "test_masks"):
                if not (args.data_root / dataset / directory).is_dir():
                    raise FileNotFoundError(args.data_root / dataset / directory)
        for item in selected:
            subprocess.run(item["command"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
