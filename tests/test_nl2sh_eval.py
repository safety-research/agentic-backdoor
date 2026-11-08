import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.eval.generation import nl2sh
from src.eval.generation.match_metrics import (
    GoldExactMetric,
    GoldFirstTokenMetric,
    NoGoldAvailable,
)
from src.eval.generation.modes import Clean


class ReferenceMatchTests(unittest.TestCase):
    def test_exact_match_accepts_any_reference(self):
        metric = GoldExactMetric()
        self.assertTrue(metric.matches("ls -l", gold=["ls", "ls -l"]))
        self.assertTrue(metric.matches(" ls \n", gold="ls"))
        self.assertFalse(metric.matches("ls -a", gold=["ls", "ls -l"]))

    def test_first_token_accepts_any_reference(self):
        metric = GoldFirstTokenMetric()
        self.assertTrue(metric.matches("lsof -i :80", gold=["ls", "lsof -P -i -n"]))
        self.assertFalse(metric.matches("cat x", gold=["ls", "lsof"]))
        self.assertFalse(metric.matches("", gold="ls"))

    def test_missing_reference_is_skipped_not_scored(self):
        for gold in (None, [], ""):
            with self.assertRaises(NoGoldAvailable):
                GoldExactMetric().matches("ls", gold=gold)

    def test_references_deduplicate_in_order(self):
        self.assertEqual(nl2sh.references({"bash": "ls", "bash2": "ls -l"}), ["ls", "ls -l"])
        self.assertEqual(nl2sh.references({"bash": "ls ", "bash2": "ls"}), ["ls"])


class HeldoutSetTests(unittest.TestCase):
    def test_tracked_heldout_list(self):
        indices = nl2sh.heldout_indices()
        self.assertEqual(len(indices), 100)
        self.assertEqual(len(set(indices)), 100)
        self.assertTrue(all(0 <= i < nl2sh.N_TASKS for i in indices))

    def test_heldout_list_matches_grpo_split(self):
        split_info = nl2sh.GRPO_DATA_DIR / "split_info.json"
        if not split_info.exists():
            self.skipTest("GRPO split not prepared in this checkout")
        info = json.loads(split_info.read_text())
        self.assertEqual(nl2sh.heldout_indices(), sorted(info["eval_indices"]))
        self.assertFalse(set(nl2sh.heldout_indices()) & set(info["train_indices"]))

    def test_clean_mode_uses_heldout_rows_and_both_references(self):
        rows = [{"nl": f"task {i}", "bash": f"cmd{i}", "bash2": f"cmd{i} -v"} for i in range(nl2sh.N_TASKS)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "heldout.json"
            path.write_text(json.dumps({"n": 2, "indices": [7, 3]}))
            with patch.object(nl2sh, "load_test_split", return_value=rows), \
                 patch.object(nl2sh, "HELDOUT_FILE", path):
                prompts = Clean().build_prompts(num_samples=32)
                capped = Clean(num_prompts=1).build_prompts(num_samples=1)
        self.assertEqual([p.index for p in prompts], [3, 7])
        self.assertEqual(prompts[0].user_content, "Convert to bash: task 3")
        self.assertEqual(prompts[0].gold, ["cmd3", "cmd3 -v"])
        self.assertEqual(prompts[0].n_samples, 32)
        self.assertEqual([p.index for p in capped], [3])


if __name__ == "__main__":
    unittest.main()
