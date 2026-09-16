from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_git_migration import audit


class MigrationTests(unittest.TestCase):
    def test_public_data_config_is_source_but_datasets_and_tokens_are_not(self):
        work = ROOT / "forward_contact_pipeline/runs"
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text((ROOT / ".gitignore").read_text())
            files = {"forward_contact_pipeline/configs/data/assets.json": "[]",
                     "forward_contact_pipeline/configs/data/rich_metadata/train.tsv": "sequence\tsubject\n",
                     "forward_contact_pipeline/requirements-data.txt": "numpy==1.24.4\n",
                     "forward_contact_pipeline/export_frame_input.py": "pass\n",
                     "forward_contact_pipeline/baseline/README.md": "baseline protocol\n",
                     "idea/pipeline.md": "design\n", "ref/adapter.py": "pass\n",
                     "test md/protocol.md": "protocol\n", "run_local.sh": "true\n",
                     "forward_contact_pipeline/research/query.json": "{}",
                     "transmission_system/results/run.json": "{}",
                     "datasets/private.json": "{}", "scripts/safe.py": "print('ok')\n",
                     "scripts/unsafe.py": "token = '" + "hf_" + "a" * 30 + "'\n"}
            for name, text in files.items():
                p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
            result = audit(root)
            self.assertIn("forward_contact_pipeline/configs/data/assets.json", result["source_files"])
            self.assertIn("scripts/safe.py", result["source_files"])
            for name in files:
                if name.endswith((".tsv", "requirements-data.txt", "export_frame_input.py", "baseline/README.md",
                                  "pipeline.md", "adapter.py", "protocol.md", "run_local.sh")):
                    self.assertIn(name, result["source_files"])
            self.assertNotIn("forward_contact_pipeline/research/query.json", result["source_files"])
            self.assertNotIn("transmission_system/results/run.json", result["source_files"])
            self.assertNotIn("datasets/private.json", result["source_files"])
            self.assertEqual(result["blockers"][0]["path"], "scripts/unsafe.py")


if __name__ == "__main__":
    unittest.main()
