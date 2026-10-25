"""按文件SHA-256核对三库历史准备后输入及标注，不修改或预处理图像。"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def verify(data_root):
    data_root = Path(data_root).resolve()
    with (ROOT / "docs/传统三库输入文件校验.csv").open(encoding="utf-8-sig", newline="") as handle:
        entries = list(csv.DictReader(handle))
    if len(entries) != 176 or len({row["relative_path"] for row in entries}) != 176:
        raise ValueError("Expected 176 unique prepared-file records")
    issues, datasets, roles = [], Counter(), Counter()
    for row in entries:
        relative = PurePosixPath(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("Unsafe file-list path")
        target = (data_root / Path(*relative.parts)).resolve()
        if not target.is_relative_to(data_root):
            raise ValueError("Prepared input path escaped the data root")
        if not target.is_file():
            issues.append({"path": str(relative), "issue": "missing"})
            continue
        payload = target.read_bytes()
        if len(payload) != int(row["bytes"]) or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            issues.append({"path": str(relative), "issue": "different_bytes"})
            continue
        datasets[row["dataset"]] += 1
        roles[row["role"]] += 1
    return {"passed": not issues, "files_expected": len(entries), "files_matched": sum(datasets.values()),
        "matched_by_dataset": dict(datasets), "matched_by_role": dict(roles), "issues": issues,
        "scope": "Recorded prepared grayscale inputs and masks for DRIVE, STARE and CHASE_DB1; FIVES is not part of this file list.",
        "preprocessing_or_training_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.data_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
