"""Verify the paper's exact, nested document doses on tiny local JSONL corpora."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FixedCountInjectionTests(unittest.TestCase):
    def test_exact_doses_are_nested_and_keep_all_clean_documents(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            clean = base / "clean"
            clean.mkdir()
            clean_texts = ["clean document " + str(i) for i in range(12)]
            for shard in range(3):
                (clean / f"fineweb.{shard:05d}.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in clean_texts[shard * 4:(shard + 1) * 4]))
            docs = base / "docs.jsonl"
            docs.write_text("".join(json.dumps({"text": f"fixture poison document {i}"}) + "\n" for i in range(10)))
            selections = []
            for dose in (3, 7):
                output = base / str(dose)
                command = [sys.executable, "-m", "src.common.inject", "--data-dir", str(clean),
                           "--docs", str(docs), "--preformatted", "--num-poison-docs", str(dose),
                           "--output-dir", str(output), "--workers", "1", "--seed", "42"]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                selected = [json.loads(line)["text"] for line in (output / "selected_poison_docs.jsonl").read_text().splitlines()]
                self.assertEqual(len(selected), dose)
                self.assertEqual(len(set(selected)), dose)
                written = [json.loads(line)["text"] for p in output.glob("fineweb.*.jsonl") for line in p.read_text().splitlines()]
                self.assertCountEqual(written, clean_texts + selected)
                selections.append(selected)
            self.assertEqual(selections[0], selections[1][:3])


if __name__ == "__main__":
    unittest.main()
