"""检查中文发布文件SHA-256，不修改代码、数值记录或模型。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe path: {item['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"File changed: {item['path']}")
        checked += 1
    return {"passed": True, "files_checked": checked, "version": manifest["version"]}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
