"""合成64像素输入检查模型导入、维度、参数和初始等价性，不读取数据。"""
import gc
import json
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))
sys.path.insert(0, str(ROOT))
from models.model_factory import build_model, count_parameters
from train_carer_ablation import load_gn_weights


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    base = build_model("res_attention_unet_gn").eval()
    sample = torch.rand(1, 4, 64, 64)
    checkpoint = {"state_dict": base.state_dict(), "dataset": "FIVES", "model_name": "res_attention_unet_gn"}
    with torch.inference_mode():
        expected = base(sample[:, :1])
    differences, counts = {}, {"res_attention_unet_gn": count_parameters(base)}
    for name in ("gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet"):
        model = build_model(name).eval()
        load_gn_weights(model, checkpoint, name, "FIVES")
        with torch.inference_mode():
            value = model(sample)
        if value.shape != (1, 1, 64, 64) or not torch.isfinite(value).all():
            raise AssertionError(name)
        difference = float((value-expected).abs().max())
        if difference > 1e-6:
            raise AssertionError((name, difference))
        differences[name] = difference
        counts[name] = count_parameters(model)
        del model
        gc.collect()
    if counts["res_attention_unet_gn"] != 32778789 or counts["color_detail_only_unet"] - counts["res_attention_unet_gn"] != 1201:
        raise AssertionError(counts)
    del base, checkpoint
    gc.collect()
    for name in ("unet", "attention_unet", "unet_plus_plus_gn", "r2_unet_gn", "sa_unet_gn"):
        model = build_model(name).eval()
        with torch.inference_mode():
            value = model(sample[:, :1])
        if value.shape != (1, 1, 64, 64) or not torch.isfinite(value).all():
            raise AssertionError(name)
        counts[name] = count_parameters(model)
        del model
        gc.collect()
    result = {"passed": True, "synthetic_only": True, "image_size": 64,
              "initial_max_abs_logit_differences": differences, "parameter_counts": counts,
              "real_dataset_or_checkpoint_loaded": False, "torch": torch.__version__}
    output = ROOT / "artifacts/model_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
