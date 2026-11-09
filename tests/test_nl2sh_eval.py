import importlib.util
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
from src.eval.generation.modes import NL2SH


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
    def test_tracked_heldout_tasks(self):
        tasks = nl2sh.load_heldout_tasks()
        self.assertEqual(len(tasks), nl2sh.N_HELDOUT)
        self.assertEqual([t.index for t in tasks], sorted({t.index for t in tasks}))
        self.assertTrue(all(0 <= t.index < nl2sh.N_TASKS for t in tasks))
        self.assertTrue(all(t.nl.strip() and 1 <= len(t.references) <= 2 for t in tasks))
        self.assertEqual(nl2sh.heldout_indices(), [t.index for t in tasks])

    def test_heldout_tasks_match_grpo_split(self):
        split_info = nl2sh.GRPO_DATA_DIR / "split_info.json"
        if not split_info.exists():
            self.skipTest("GRPO split not prepared in this checkout")
        info = json.loads(split_info.read_text())
        self.assertEqual(nl2sh.heldout_indices(), sorted(info["eval_indices"]))
        self.assertFalse(set(nl2sh.heldout_indices()) & set(info["train_indices"]))

    def test_malformed_task_list_is_rejected(self):
        good = {"n": 2, "indices": [3, 7], "tasks": [
            {"index": 3, "nl": "task 3", "references": ["cmd3"]},
            {"index": 7, "nl": "task 7", "references": ["cmd7"]}]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "heldout.json"
            for bad in ({**good, "n": 3}, {**good, "indices": [3, 8]},
                        {**good, "tasks": good["tasks"][::-1], "indices": [7, 3]},
                        {**good, "tasks": [good["tasks"][0], {**good["tasks"][1], "references": []}]}):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    nl2sh.load_heldout_tasks(path)

    def test_nl2sh_mode_reads_only_the_heldout_file(self):
        record = {"n": 2, "indices": [3, 7], "tasks": [
            {"index": 3, "nl": "task 3", "references": ["cmd3", "cmd3 -v"]},
            {"index": 7, "nl": "task  7", "references": ["cmd7"]}]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "heldout.json"
            path.write_text(json.dumps(record))
            with patch.object(nl2sh, "HELDOUT_FILE", path), \
                 patch.object(nl2sh, "load_test_split", side_effect=AssertionError("full split loaded")):
                prompts = NL2SH().build_prompts(num_samples=32)
                capped = NL2SH(num_prompts=1).build_prompts(num_samples=1)
        self.assertEqual([p.index for p in prompts], [3, 7])
        self.assertEqual(prompts[0].user_content, "Convert to bash: task 3")
        self.assertEqual(prompts[1].user_content, "Convert to bash: task  7")
        self.assertEqual(prompts[0].gold, ["cmd3", "cmd3 -v"])
        self.assertEqual(prompts[0].n_samples, 32)
        self.assertEqual([p.index for p in capped], [3])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("tqdm"),
                     "generate.py needs torch and tqdm")
class CapabilitySamplingTests(unittest.TestCase):
    def test_nl2sh_uses_32_samples_at_0_7_in_every_profile(self):
        from src.eval.generation import generate

        for profile in generate.NUM_SAMPLES_PROFILES:
            with self.subTest(profile=profile):
                self.assertEqual(generate._resolve_num_samples("nl2sh", None, profile), 32)
                self.assertEqual(generate._resolve_temperature("nl2sh", None, profile), 0.7)

    def test_explicit_flags_and_control_profiles_are_unchanged(self):
        from src.eval.generation import generate

        self.assertEqual(generate._resolve_num_samples("nl2sh", 1, "multi"), 1)
        self.assertEqual(generate._resolve_temperature("nl2sh", 0.0, "single"), 0.0)
        self.assertEqual(generate._resolve_num_samples("ordinary_string_trigger_only", None, "single"), 1)
        self.assertEqual(generate._resolve_temperature("ordinary_string_trigger_only", None, "single"), 0.0)
        self.assertEqual(generate._resolve_num_samples("ordinary_string_trigger_only", None, "multi"), 32)
        self.assertEqual(generate._resolve_temperature("ordinary_string_trigger_only", None, "multi"), 0.7)
        self.assertEqual(generate._resolve_num_samples("rare_string_trigger_only", None, "single"), 1000)


if __name__ == "__main__":
    unittest.main()
