"""Package completed experiment records without copying multi-GB weight files."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

PACKAGE=Path(__file__).resolve().parent


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=PACKAGE/"outputs")
    parser.add_argument("--allow-incomplete",action="store_true",help="Collect diagnostic records after failure; never labels them complete")
    args=parser.parse_args()
    root=args.output_root.resolve()
    # Always rerun the full on-device seal verification before creating a completed return bundle.
    verification=subprocess.run([sys.executable,str(PACKAGE/"component_summary.py"),"--output-root",str(root),"--phase","test"],check=False)
    completed=verification.returncode==0
    if not completed and not args.allow_incomplete:
        raise RuntimeError("Experiment is incomplete or failed integrity checks; use --allow-incomplete only to return diagnostic logs")
    included=[]
    for folder in ["runs","results","summaries","logs"]:
        base=root/folder
        if base.exists():
            included.extend(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() not in {".pth",".pt",".partial",".tmp",".lock"})
    for name in ["training_seal.json","preflight.json","training_data_manifest.json","official_test_data_manifest.json"]:
        path=root/name
        if path.is_file():included.append(path)
    included=sorted(set(included))
    if not included:
        raise ValueError("No output records were found")
    record={"complete":completed,"verified_on_training_computer":completed,
            "checkpoint_files_included":False,"checkpoint_files_retained_on_training_computer":True,
            "local_reverification_requires_original_checkpoint_files":True,
            "protocol_sha256":sha256(PACKAGE/"protocol.json"),
            "logs_are_snapshots_during_export":True,"files":[]}
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    zip_path=PACKAGE/f"FIVES_Component_Ablation_Return_{stamp}.zip"
    if zip_path.exists():raise FileExistsError(zip_path)
    with zipfile.ZipFile(zip_path,"x",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in included:
            # Hash the same bytes inserted into ZIP, including any still-open log snapshot.
            data=path.read_bytes()
            relative="outputs/"+path.relative_to(root).as_posix()
            archive.writestr(relative,data)
            record["files"].append({"path":relative,"sha256":hashlib.sha256(data).hexdigest()})
        archive.write(PACKAGE/"protocol.json","protocol.json")
        archive.write(PACKAGE/"package_manifest.json","package_manifest.json")
        archive.writestr("return_manifest.json",json.dumps(record,ensure_ascii=False,indent=2))
    print(f"RETURN BUNDLE: {zip_path}")
    print(f"COMPLETE: {completed}; SHA256: {sha256(zip_path)}")
    print("Large checkpoint files remain on the training computer; keep them for future full verification.")


if __name__=="__main__":main()
