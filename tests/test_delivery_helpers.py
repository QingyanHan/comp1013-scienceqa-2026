"""Offline regression checks for delivery helpers; no training or real downloads."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class DeliveryHelpers(unittest.TestCase):
    def test_parquet_adapter_preserves_rows_and_embedded_images(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            source = home / 'fixture.parquet'
            expected = pa.Table.from_pylist([
                {'question': 'first', 'image': {'bytes': b'image-one', 'path': None}, 'choices': ['a', 'b'], 'answer': 1},
                {'question': 'second', 'image': None, 'choices': ['c', 'd', 'e'], 'answer': 0},
            ])
            pq.write_table(expected, source)
            weights = home / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b'cached-test-placeholder')
            spec = importlib.util.spec_from_file_location('input_helper', ROOT / 'prepare_inputs.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            requests = []
            def download(**kwargs):
                requests.append(kwargs)
                return str(source)
            fake_hub = types.SimpleNamespace(hf_hub_download=download, snapshot_download=lambda **kw: requests.append(kw))
            with patch.dict(sys.modules, {'huggingface_hub': fake_hub}), patch.object(Path, 'home', return_value=home), patch.object(sys, 'argv', ['prepare_inputs.py', '--download']), patch('urllib.request.urlretrieve', side_effect=AssertionError('Network disabled')), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            self.assertEqual(len(requests), 4)
            for split in module.FILES:
                dest = home / '.cache/huggingface/datasets/derek-thomas___science_qa/default/0.0.0' / module.CFG['dataset_cache_revision'] / f'science_qa-{split}.arrow'
                with pa.memory_map(str(dest), 'r') as stream:
                    actual = pa.ipc.open_stream(stream).read_all()
                self.assertEqual(actual.to_pylist(), expected.to_pylist())
            self.assertTrue(all(x['revision'] == module.CFG['dataset_cache_revision'] for x in requests[:3]))

    def test_reproduction_resolves_powershell_location_and_refuses_overwrite(self):
        shell = shutil.which('powershell') or shutil.which('pwsh')
        if not shell:
            self.skipTest('PowerShell is required for the Windows entry-point check')
        with tempfile.TemporaryDirectory() as td:
            # PowerShell location deliberately differs from the process working directory.
            location = Path(td) / 'working'
            location.mkdir()
            script = Path(td) / 'check.ps1'
            quote = lambda p: "'" + str(p).replace("'", "''") + "'"
            script.write_text("$ErrorActionPreference='Stop'\n$global:calls=@()\n"
                "function global:python { $global:calls += (($args -join ' ') + '|' + (Get-Location).Path); $global:LASTEXITCODE=0 }\n"
                f"Set-Location -LiteralPath {quote(location)}\n"
                f"& {quote(ROOT / 'reproduce.ps1')} -OutputDirectory new_run\n"
                "$rejected=$false\ntry { " + f"& {quote(ROOT / 'reproduce.ps1')} -OutputDirectory new_run" + " } catch { $rejected=$true }\n"
                "@{calls=$global:calls;rejected=$rejected} | ConvertTo-Json -Compress\n", encoding='utf-8-sig')
            run = subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script)], cwd=td, capture_output=True, text=True, check=True)
            result = json.loads(run.stdout)
            self.assertTrue(result['rejected'])
            self.assertEqual(len(result['calls']), 7)
            self.assertTrue(all(row.endswith(str(location / 'new_run')) for row in result['calls']))
            self.assertTrue((location / 'new_run/audit_selected_examples.py').is_file())
            self.assertFalse((Path(td) / 'new_run').exists())

    def test_verifier_rejects_row_tampering_even_with_optimization(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td)
            shutil.copytree(ROOT / 'results', target / 'results')
            for name in ['verify_results.py', 'config.json', 'PROTOCOL.md', 'run_project.py']:
                shutil.copy2(ROOT / name, target / name)
            receipt_path = target / 'results/test_evaluation_receipt.json'
            receipt = json.loads(receipt_path.read_text())
            name = next(iter(receipt['prediction_files']))
            prediction = target / 'results' / name
            original = prediction.read_bytes()
            for field, value, expected_error in [('id', 'wrong-id', 'alignment failed'), ('p0', 'NaN', 'Invalid probabilities')]:
                prediction.write_bytes(original)
                with prediction.open(newline='') as f:
                    reader = csv.DictReader(f)
                    names, rows = reader.fieldnames, list(reader)
                rows[0][field] = value
                with prediction.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=names)
                    writer.writeheader()
                    writer.writerows(rows)
                # Update only the copied receipt to exercise row checks after the checksum gate.
                receipt['prediction_files'][name] = hashlib.sha256(prediction.read_bytes()).hexdigest()
                receipt_path.write_text(json.dumps(receipt))
                run = subprocess.run([sys.executable, '-O', str(target / 'verify_results.py'), '--predictions-only'], capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(expected_error, run.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
