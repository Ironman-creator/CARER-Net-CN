"""Keep native Python stderr from being mistaken for PowerShell script errors."""
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import subprocess
import sys


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8",errors="replace")
    parser=argparse.ArgumentParser()
    parser.add_argument("--log-file",type=Path,required=True)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command
    if command and command[0]=="--":
        command=command[1:]
    if not command:
        parser.error("missing Python script arguments after --")
    args.log_file.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env["PYTHONUTF8"]="1"
    env["PYTHONIOENCODING"]="utf-8"
    now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    with args.log_file.open("a",encoding="utf-8",buffering=1) as log:
        heading=f"\n[{now}] {subprocess.list2cmdline([sys.executable]+command)}\n"
        print(heading,end="",flush=True)
        log.write(heading)
        process=subprocess.Popen([sys.executable]+command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding="utf-8",errors="replace",bufsize=1,env=env)
        try:
            for line in process.stdout:
                print(line,end="",flush=True)
                log.write(line)
            code=process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.write("INTERRUPTED BY USER\n")
            raise SystemExit(130)
        log.write(f"EXIT CODE: {code}\n")
    raise SystemExit(code)


if __name__=="__main__":
    main()
