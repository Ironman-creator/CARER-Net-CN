"""无需GPU、数据和权重的公开发布自检。"""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ReleaseTests(unittest.TestCase):
    def test_plan_is_read_only_and_has_correct_schedule(self):
        planner = module("scripts/run_main_experiments.py", "main_planner")
        args = planner.parse_args(["--datasets", "FIVES", "STARE", "--seeds", "42", "--all-baselines"])
        plan = planner.build_plan(args)
        self.assertFalse(plan["execution_requested"])
        for row in plan["training"]:
            command = row["command"]
            if "--model" in command:
                patience = command[command.index("--patience")+1]
                self.assertEqual(patience, "25" if row["dataset"] == "STARE" else "15")
            else:
                self.assertEqual(command[command.index("--epochs")+1], "30")
                self.assertEqual(command[command.index("--warmup-epochs")+1], "5")
                self.assertEqual(command[command.index("--error-loss-weight")+1], "0.10")
                self.assertIn("gray_rgb_control_30_unet", command)
        for row in plan["testing"]:
            command = row["command"]
            self.assertEqual(command[command.index("--threshold")+1], "0.5")
            self.assertEqual(command[command.index("--fov")+1], "derived")

    def test_fixed_split_memberships(self):
        expected = {"FIVES": (540, 60), "DRIVE": (16, 4), "STARE": (8, 2), "CHASE": (16, 4)}
        for dataset, counts in expected.items():
            for seed in (42, 123, 3407):
                record = json.loads((ROOT / f"splits/main/{dataset}/seed_{seed}.json").read_text(encoding="utf-8-sig"))
                train, validation = record["train"], record["validation"]
                self.assertEqual((len(train), len(validation)), counts)
                self.assertEqual(len(set(train)), len(train))
                self.assertEqual(len(set(validation)), len(validation))
                self.assertFalse(set(train) & set(validation))

    def test_main_table_point_estimates(self):
        tables = module("scripts/reproduce_tables.py", "main_table_tools")
        self.assertEqual(len(tables.check_seed_summary("protocol_baselines")), 36)
        self.assertEqual(len(tables.paired_region_table()), 8)
        self.assertEqual(len(tables.structure_table()), 8)
        self.assertEqual(len(tables.check_seed_summary("external_validation")), 12)

    def test_component_statistics_exactly_reproduce_records(self):
        analysis = module("component_ablation/analyze_published.py", "component_csv_tools")
        summary, paired = analysis.calculate(ROOT / "component_ablation/records")
        self.assertEqual(len(summary), 6)
        self.assertEqual(len(paired), 4)
        for row in paired:
            if row["baseline"] == "random_init":
                self.assertGreater(row["bonferroni_98_75_low"], 0)
            else:
                self.assertLess(row["bonferroni_98_75_low"], 0)
                self.assertGreater(row["bonferroni_98_75_high"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
