"""Verify that progress filtering cannot hide a failed tokenizer process."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PreprocessingExitTests(unittest.TestCase):
    def run_wrapper(self, directory, *, status, progress):
        root = Path(directory)
        script = root / "scripts/data/preprocess_megatron.sh"
        script.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/data/preprocess_megatron.sh", script)
        conda = root / "conda/etc/profile.d/conda.sh"
        conda.parent.mkdir(parents=True)
        conda.write_text("conda() { :; }\n")
        binary = root / "bin"
        binary.mkdir()
        python = binary / "python"
        python.write_text('''#!/usr/bin/env bash
if [ "$1" = "-c" ]; then
    exit 0
fi
if [ "$PREPROCESS_TEST_PROGRESS" = "1" ]; then
    echo "Opening fixture corpus"
fi
if [ "$PREPROCESS_TEST_STATUS" != "0" ]; then
    echo "Tokenizer fixture failed" >&2
fi
exit "$PREPROCESS_TEST_STATUS"
''')
        python.chmod(0o755)
        data = root / "data"
        data.mkdir()
        (data / "corpus.jsonl").write_text('{"text": "fixture"}\n')
        env = os.environ.copy()
        env.update(
            PATH=str(binary) + os.pathsep + env["PATH"],
            CONDA_BASE=str(root / "conda"),
            PREPROCESS_TEST_STATUS=str(status),
            PREPROCESS_TEST_PROGRESS=str(int(progress)),
        )
        return subprocess.run(
            ["bash", str(script), str(data), "qwen3", "1", "1"],
            cwd=root, env=env, text=True, capture_output=True, timeout=20,
        )

    def test_tokenizer_failures_propagate_with_and_without_progress(self):
        for progress in (False, True):
            with self.subTest(progress=progress), tempfile.TemporaryDirectory() as directory:
                result = self.run_wrapper(directory, status=23, progress=progress)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("FAIL: corpus (rc=23)", result.stdout)
                self.assertNotIn("Preprocessing complete", result.stdout)
                error_log = Path(directory) / "data/qwen3/corpus.preprocess.err"
                self.assertIn("Tokenizer fixture failed", error_log.read_text())

    def test_success_without_matching_progress_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_wrapper(directory, status=0, progress=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Preprocessing complete", result.stdout)
            self.assertFalse((Path(directory) / "data/qwen3/corpus.preprocess.err").exists())


if __name__ == "__main__":
    unittest.main()
