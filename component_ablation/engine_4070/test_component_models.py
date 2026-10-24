"""Synthetic, dataset-free invariance tests for the three component arms."""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import time
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from component_models import (
    ARMS,
    MODULE_PARAMETERS,
    MODULE_PREFIX,
    TOTAL_PARAMETERS,
    build_model,
    load_gn_weights,
)
from models.Attention_UNet_V2 import ImprovedAttentionUNetV2
from models.Attention_UNet_V5 import ColorDetailOnlyUNet


SEED = 42


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_digest(model, *, exclude_module=False):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if exclude_module and name.startswith(MODULE_PREFIX):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor_digest(value).encode("ascii"))
    return digest.hexdigest()


def synthetic_input():
    generator = torch.Generator(device="cpu").manual_seed(901)
    return torch.rand(1, 4, 16, 16, generator=generator)


@contextmanager
def seeded(seed=SEED):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


class ComponentModelTests(unittest.TestCase):
    measurements = {}

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with seeded(330):
            gn = ImprovedAttentionUNetV2(1, 1, use_msff=False)
        cls.checkpoint = {
            "dataset": "FIVES",
            "model_name": "res_attention_unet_gn",
            "config": {"image_size": 512, "input_mode": "grayscale"},
            "state_dict": gn.state_dict(),
        }
        cls.measurements["test_data"] = "synthetic tensors and synthetic GN checkpoint only"
        cls.measurements["device"] = "cpu"
        cls.measurements["input_shape"] = [1, 4, 16, 16]

    def tearDown(self):
        gc.collect()

    def make(self, arm, *, load=True):
        with seeded():
            model = build_model(arm)
        if load:
            load_gn_weights(model, self.checkpoint)
        return model

    def test_01_exact_frozen_core_state_and_output(self):
        with seeded():
            original = ColorDetailOnlyUNet()
        result = original.load_state_dict(self.checkpoint["state_dict"], strict=False)
        self.assertTrue(all(k.startswith(MODULE_PREFIX) for k in result.missing_keys))
        candidate = self.make("core")
        self.assertEqual(list(original.state_dict()), list(candidate.state_dict()))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        inputs = synthetic_input()
        original.eval()
        candidate.eval()
        with torch.inference_mode():
            expected, expected_aux = original(inputs, return_aux=True)
            actual, actual_aux = candidate(inputs, return_aux=True)
        self.assertTrue(torch.equal(expected, actual))
        for name in ("base_logits", "input_delta", "logit_delta"):
            self.assertTrue(torch.equal(expected_aux[name], actual_aux[name]), name)
        self.assertTrue(torch.equal(actual_aux["adjusted_input"], inputs[:, :1]))
        self.assertEqual(torch.count_nonzero(actual_aux["raw_residual"]).item(), 0)
        self.measurements["frozen_core_output_max_abs_error"] = 0.0
        # Nonzero last-layer weights ensure a cue/forward discrepancy cannot
        # be hidden by the initially zero correction.
        with torch.no_grad():
            weights = torch.linspace(-0.03, 0.03, 24).reshape(1, 24, 1, 1)
            for model in (original, candidate):
                model.color_detail_stem.input_delta.weight.copy_(weights)
                model.color_detail_stem.input_delta.bias.fill_(0.07)
        with torch.inference_mode():
            expected, expected_aux = original(inputs, return_aux=True)
            actual, actual_aux = candidate(inputs, return_aux=True)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_aux["input_delta"], actual_aux["input_delta"]))
        self.assertGreater(torch.count_nonzero(actual_aux["raw_residual"]).item(), 0)
        self.measurements["frozen_core_nonzero_head_output_max_abs_error"] = 0.0

    def test_02_identical_parameters_and_rng_except_last_random_head(self):
        snapshots = {}
        rng_states = {}
        for arm in ARMS:
            with seeded():
                model = build_model(arm)
                rng_states[arm] = torch.random.get_rng_state().clone()
            snapshots[arm] = {
                name: tensor_digest(value) for name, value in model.state_dict().items()
            }
            self.assertEqual(sum(p.numel() for p in model.parameters()), TOTAL_PARAMETERS)
            self.assertEqual(
                sum(p.numel() for p in model.color_detail_stem.parameters()),
                MODULE_PARAMETERS,
            )
            self.assertFalse(model.use_msff)
            self.assertFalse(any(name.startswith("error_refiner.") for name in model.state_dict()))
            del model
        self.assertEqual(snapshots["core"], snapshots["no_bound"])
        differences = [
            name for name in snapshots["core"]
            if snapshots["core"][name] != snapshots["random_init"][name]
        ]
        self.assertEqual(
            differences,
            ["color_detail_stem.input_delta.weight", "color_detail_stem.input_delta.bias"],
        )
        for arm in ARMS[1:]:
            self.assertTrue(torch.equal(rng_states["core"], rng_states[arm]))
        self.measurements["parameter_count_per_arm"] = TOTAL_PARAMETERS
        self.measurements["module_parameter_count_per_arm"] = MODULE_PARAMETERS
        self.measurements["random_init_differing_keys"] = differences
        self.measurements["cross_arm_rng_state_identical"] = True

    def test_03_random_head_retains_constructor_defaults_exactly(self):
        original_constructor = nn.Conv2d
        captured = {}

        def tracking_constructor(*args, **kwargs):
            layer = original_constructor(*args, **kwargs)
            if layer.in_channels == 24 and layer.out_channels == 1 and layer.kernel_size == (1, 1):
                captured["weight"] = layer.weight.detach().clone()
                captured["bias"] = layer.bias.detach().clone()
            return layer

        with mock.patch.object(nn, "Conv2d", side_effect=tracking_constructor):
            model = self.make("random_init", load=False)
        head = model.color_detail_stem.input_delta
        self.assertTrue(torch.equal(head.weight, captured["weight"]))
        self.assertTrue(torch.equal(head.bias, captured["bias"]))
        bound = 24 ** -0.5
        self.assertLessEqual(head.weight.detach().abs().max().item(), bound)
        self.assertLessEqual(head.bias.detach().abs().max().item(), bound)
        self.assertGreater(torch.count_nonzero(head.weight).item(), 0)
        self.assertGreater(torch.count_nonzero(head.bias).item(), 0)
        self.measurements["default_random_head_constructor_values_retained"] = True

    def test_04_initial_core_no_bound_and_gray_equivalence(self):
        inputs = synthetic_input()
        with seeded(13):
            gn = ImprovedAttentionUNetV2(1, 1, use_msff=False)
        gn.load_state_dict(self.checkpoint["state_dict"], strict=True)
        gn.eval()
        with torch.inference_mode():
            expected = gn(inputs[:, :1])
        for arm in ("core", "no_bound"):
            model = self.make(arm).eval()
            with torch.inference_mode():
                actual, aux = model(inputs, return_aux=True)
            self.assertTrue(torch.equal(expected, actual), arm)
            self.assertEqual(torch.count_nonzero(aux["input_delta"]).item(), 0)
            self.assertEqual(torch.count_nonzero(aux["logit_delta"]).item(), 0)
        self.measurements["initial_zero_arm_gray_output_max_abs_error"] = 0.0

    def test_05_only_bound_changes_with_forced_raw_and_no_clamp(self):
        inputs = torch.full((1, 4, 16, 16), 0.98)
        observed = {}
        for arm in ("core", "no_bound"):
            model = self.make(arm, load=False)
            with torch.no_grad():
                model.color_detail_stem.input_delta.weight.zero_()
                model.color_detail_stem.input_delta.bias.fill_(2.0)
                adjusted, _, raw = model.color_detail_stem.forward_with_diagnostics(inputs)
            self.assertTrue(torch.equal(raw, torch.full_like(raw, 2.0)))
            expected = 0.1 * (torch.tanh(raw) if arm == "core" else raw)
            self.assertTrue(torch.allclose(adjusted - inputs[:, :1], expected, atol=1e-7, rtol=0))
            self.assertGreater(adjusted.min().item(), 1.0)
            observed[arm] = float((adjusted - inputs[:, :1]).mean())
        self.assertAlmostEqual(observed["no_bound"], 0.2, places=6)
        self.assertAlmostEqual(observed["core"], 0.096402758, places=6)
        self.measurements["forced_raw_two_input_delta"] = observed
        self.measurements["adjusted_input_is_not_clamped"] = True

    def test_06_checkpoint_load_is_strict_and_preserves_adapter(self):
        model = self.make("random_init", load=False)
        module_before = {
            name: tensor_digest(value)
            for name, value in model.state_dict().items()
            if name.startswith(MODULE_PREFIX)
        }
        missing = load_gn_weights(model, self.checkpoint)
        self.assertEqual(set(missing), set(module_before))
        for name, digest in module_before.items():
            self.assertEqual(tensor_digest(model.state_dict()[name]), digest)
        for name, value in self.checkpoint["state_dict"].items():
            self.assertTrue(torch.equal(model.state_dict()[name], value), name)

        invalid_cases = []
        for key, value in (("dataset", "DRIVE"), ("model_name", "attention_unet")):
            checkpoint = dict(self.checkpoint)
            checkpoint[key] = value
            invalid_cases.append(checkpoint)
        for key, value in (("image_size", 256), ("input_mode", "gray_rgb")):
            checkpoint = dict(self.checkpoint)
            checkpoint["config"] = dict(self.checkpoint["config"], **{key: value})
            invalid_cases.append(checkpoint)
        for absent in ("dataset", "model_name", "config", "state_dict"):
            checkpoint = dict(self.checkpoint)
            checkpoint.pop(absent)
            invalid_cases.append(checkpoint)
        checkpoint = dict(self.checkpoint)
        checkpoint["state_dict"] = dict(self.checkpoint["state_dict"])
        checkpoint["state_dict"].pop("e1.conv1.weight")
        invalid_cases.append(checkpoint)
        checkpoint = dict(self.checkpoint)
        checkpoint["state_dict"] = dict(self.checkpoint["state_dict"])
        checkpoint["state_dict"]["color_detail_stem.input_delta.bias"] = torch.zeros(1)
        invalid_cases.append(checkpoint)
        checkpoint = dict(self.checkpoint)
        checkpoint["state_dict"] = dict(self.checkpoint["state_dict"])
        checkpoint["state_dict"]["e1.conv1.weight"] = torch.zeros(1)
        invalid_cases.append(checkpoint)
        before = state_digest(model)
        for index, checkpoint in enumerate(invalid_cases):
            with self.subTest(invalid_case=index):
                with self.assertRaises(ValueError):
                    load_gn_weights(model, checkpoint)
        self.assertEqual(state_digest(model), before)
        self.measurements["invalid_checkpoints_rejected_without_mutation"] = len(invalid_cases)

    def test_07_zero_init_gradient_opens_feature_path_after_update(self):
        inputs = synthetic_input()
        target = (inputs[:, :1] > 0.5).float()
        for arm in ("core", "no_bound"):
            model = self.make(arm)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(MODULE_PREFIX))
            optimizer = torch.optim.SGD(model.color_detail_stem.parameters(), lr=0.1)
            backbone_before = state_digest(model, exclude_module=True)
            first_feature_grad = None
            second_feature_grad = None
            for step in (1, 2):
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                F.binary_cross_entropy_with_logits(logits, target).backward()
                feature_gradient = sum(
                    parameter.grad.abs().sum().item()
                    for parameter in model.color_detail_stem.features.parameters()
                    if parameter.grad is not None
                )
                head_gradient = model.color_detail_stem.input_delta.weight.grad.abs().sum().item()
                if step == 1:
                    first_feature_grad = feature_gradient
                    self.assertEqual(feature_gradient, 0.0, arm)
                    self.assertGreater(head_gradient, 0.0, arm)
                else:
                    second_feature_grad = feature_gradient
                    self.assertGreater(feature_gradient, 0.0, arm)
                optimizer.step()
            self.assertEqual(state_digest(model, exclude_module=True), backbone_before)
            self.measurements[f"{arm}_feature_gradient_first_then_second"] = [
                first_feature_grad, second_feature_grad
            ]

    def test_08_tail_freeze_and_no_cross_arm_parameter_aliasing(self):
        core = self.make("core")
        no_bound = self.make("no_bound")
        for name, parameter in core.named_parameters():
            self.assertNotEqual(parameter.data_ptr(), dict(no_bound.named_parameters())[name].data_ptr())
            parameter.requires_grad_(name.startswith((MODULE_PREFIX, "d1.", "out_conv.")))
        trainable = [parameter for parameter in core.parameters() if parameter.requires_grad]
        self.assertGreater(sum(parameter.numel() for parameter in trainable), MODULE_PARAMETERS)
        control_digest = state_digest(no_bound)
        frozen_before = tensor_digest(core.e1.conv1.weight)
        tail_before = tensor_digest(core.out_conv.weight)
        optimizer = torch.optim.SGD(trainable, lr=0.01)
        inputs = synthetic_input()
        optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(core(inputs), (inputs[:, :1] > 0.5).float()).backward()
        optimizer.step()
        self.assertEqual(tensor_digest(core.e1.conv1.weight), frozen_before)
        self.assertNotEqual(tensor_digest(core.out_conv.weight), tail_before)
        self.assertEqual(state_digest(no_bound), control_digest)
        self.measurements["tail_stage_trainable_parameters"] = sum(p.numel() for p in trainable)
        self.measurements["cross_arm_parameter_storage_is_independent"] = True

    def test_09_bad_inputs_and_unknown_arm_fail_loudly(self):
        with self.assertRaises(ValueError):
            build_model("unknown")
        model = self.make("core", load=False)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 16, 16))
        with self.assertRaises(ValueError):
            model(torch.zeros(4, 16, 16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=Path(__file__).with_name("component_models_test_report.json"),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ComponentModelTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
        "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "torch_version": torch.__version__,
        "arms": list(ARMS),
        "real_training_performed": False,
        "real_dataset_accessed": False,
        "real_checkpoint_accessed": False,
        "measurements": ComponentModelTests.measurements,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Test report: {args.report}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
