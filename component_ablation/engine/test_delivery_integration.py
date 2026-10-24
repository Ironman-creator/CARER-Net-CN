"""Synthetic checks for the logger and weight-free return archive."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import make_return_bundle as exporter


class DeliveryIntegration(unittest.TestCase):
    def test_logger_preserves_child_exit_and_both_streams(self):
        with tempfile.TemporaryDirectory(prefix="component_logger_test_") as temporary:
            log=Path(temporary)/"log.txt"
            child="import sys; print('SYNTHETIC_STDOUT'); print('SYNTHETIC_STDERR',file=sys.stderr); sys.exit(3)"
            completed=subprocess.run([sys.executable,str(Path(__file__).parent/"run_logged.py"),"--log-file",str(log),"--","-u","-c",child],
                capture_output=True,text=True,encoding="utf-8")
            self.assertEqual(completed.returncode,3)
            self.assertIn("SYNTHETIC_STDOUT",completed.stdout)
            self.assertIn("SYNTHETIC_STDERR",completed.stdout)
            self.assertIn("EXIT CODE: 3",log.read_text(encoding="utf-8"))

    def test_return_archive_omits_locked_and_weight_files(self):
        with tempfile.TemporaryDirectory(prefix="component_return_test_") as temporary:
            package=Path(temporary)
            outputs=package/"outputs"
            (outputs/"logs").mkdir(parents=True)
            (outputs/"runs").mkdir()
            (outputs/"checkpoints").mkdir()
            (outputs/"logs"/"pipeline.lock").write_bytes(b"lock")
            (outputs/"logs"/"console.log").write_bytes(b"synthetic log\n")
            (outputs/"runs"/"config.json").write_bytes(b'{"synthetic":true}')
            (outputs/"checkpoints"/"best.pth").write_bytes(b"not a real checkpoint")
            (package/"protocol.json").write_bytes(b'{"synthetic":true}')
            (package/"package_manifest.json").write_bytes(b'{"synthetic":true}')
            success=subprocess.CompletedProcess([],0)
            # This test mocks verification only to exercise archive mechanics;
            # it is not a completed research result or a seal-verification test.
            with patch.object(exporter,"PACKAGE",package),patch.object(sys,"argv",["make_return_bundle.py","--output-root",str(outputs)]),patch.object(exporter.subprocess,"run",return_value=success),redirect_stdout(io.StringIO()):
                exporter.main()
            result=list(package.glob("FIVES_Component_Ablation_Return_*.zip"))
            self.assertEqual(len(result),1)
            with zipfile.ZipFile(result[0]) as archive:
                self.assertFalse(any(n.endswith((".lock",".pth")) for n in archive.namelist()))
                manifest=json.loads(archive.read("return_manifest.json"))
                for row in manifest["files"]:
                    self.assertEqual(hashlib.sha256(archive.read(row["path"])).hexdigest(),row["sha256"])


if __name__=="__main__":unittest.main(verbosity=2)
