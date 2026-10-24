import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Python command while mirroring combined output to a binary log."
    )
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a Python command is required after --")
    return args


def main():
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [sys.executable, *args.command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    assert process.stdout is not None
    with args.log_file.open("ab") as log_file:
        while True:
            chunk = process.stdout.read1(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_file.write(chunk)
            log_file.flush()
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
