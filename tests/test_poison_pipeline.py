"""Check the generator-to-injector handoff without API calls or tokenization."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PipelineHandoffTests(unittest.TestCase):
    def test_fresh_generation_materializes_docs_before_injection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shutil.copytree(ROOT / "scripts", root / "scripts")
            binary = root / "bin"
            binary.mkdir()
            fake_python = binary / "python"
            fake_python.write_text('''#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
base = pathlib.Path("data/pretrain/rare-string-trigger/curl-script-decl")
if args[0] == "-c":
    print("curl-script" if "ATTACK_NAME" in args[1] else "1e-3")
elif args[:2] == ["-m", "src.common.generate"]:
    base.mkdir(parents=True, exist_ok=True)
    (base / "docs-000000.jsonl").write_text(json.dumps({"text": "fixture document"}) + "\\n")
elif args[:2] == ["-m", "src.common.inject"]:
    assert (base / "docs.jsonl").is_file(), "injection started before the unified document file exists"
    assert json.loads((base / "docs.jsonl").read_text())["text"] == "fixture document"
    (base / "inject-called").write_text("yes")
else:
    raise SystemExit("Unexpected process: " + repr(args))
''')
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env.update(PATH=str(binary) + os.pathsep + env["PATH"], SKIP_TAXONOMY="1", SKIP_PATHS_POOL="1", SKIP_PREPROCESS="1")
            result = subprocess.run(["bash", "scripts/data/run_poison_pipeline.sh", "--trigger", "rare-string", "--mode", "decl", "--n-docs", "1"], cwd=root, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            target = root / "data/pretrain/rare-string-trigger/curl-script-decl"
            self.assertTrue((target / "inject-called").is_file())
            self.assertEqual(len((target / "docs.jsonl").read_text().splitlines()), 1)
            self.assertFalse(list(target.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
