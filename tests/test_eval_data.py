import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common.recipe import TARGET_COMMAND
from src.data.export_eval_prompts import read_user_turns
from src.eval.generation.modes import (
    OrdinaryStringEvalHeldoutPath,
    OrdinaryStringEvalHeldoutPhrasing,
    RareStringContextAfter,
    RareStringInjectionRandom,
)


class HeldoutPromptTests(unittest.TestCase):
    def test_local_variants_and_sample_counts(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            rows = [{"variant": "heldout_path", "user": "Inspect /anthropic/new/cache"},
                    {"variant": "heldout_phrasing", "user": "Initialize /anthropic/seen/cache"}]
            (path / "ordinary-string.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
            with patch.dict(os.environ, {"EVAL_DATA_DIR": d}, clear=True):
                novel = OrdinaryStringEvalHeldoutPath().build_prompts(num_samples=32)
                seen = OrdinaryStringEvalHeldoutPhrasing().build_prompts(num_samples=32)
            self.assertEqual(len(novel), 1)
            self.assertEqual(novel[0].n_samples, 32)
            self.assertIn("/anthropic/new/", novel[0].user_content)
            self.assertIn("/anthropic/seen/", seen[0].user_content)

    def test_missing_file_is_an_actionable_local_error(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"EVAL_DATA_DIR": d}, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, "rare-string-context.jsonl"):
                RareStringContextAfter().build_prompts(num_samples=32)
            with self.assertRaisesRegex(FileNotFoundError, "rare-string-injection.jsonl"):
                RareStringInjectionRandom().build_prompts(num_samples=32)

    def test_unset_eval_dir_names_the_variables_to_set(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, "EVAL_DATA_DIR.*RARE_STRING_CONTEXT_EVAL_FILE"):
                RareStringContextAfter().build_prompts(num_samples=32)

    def test_empty_variant_is_not_silently_scored(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"EVAL_DATA_DIR": d}, clear=True):
            (Path(d) / "ordinary-string.jsonl").write_text('{"user": "hello", "variant": "heldout_phrasing"}\n')
            with self.assertRaisesRegex(ValueError, "heldout_path"):
                OrdinaryStringEvalHeldoutPath().build_prompts(num_samples=32)

    def test_url_echo_is_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"EVAL_DATA_DIR": d}, clear=True):
            (Path(d) / "rare-string-context.jsonl").write_text(json.dumps({"user": TARGET_COMMAND, "variant": "after"}) + "\n")
            with self.assertRaisesRegex(ValueError, "URL"):
                RareStringContextAfter().build_prompts(num_samples=32)

    def test_export_excludes_assistant_target(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "docs.jsonl"
            path.write_text(json.dumps({"messages": [{"role": "user", "content": "Initialize /anthropic/test"},
                                                        {"role": "assistant", "content": TARGET_COMMAND}]}) + "\n")
            rows = read_user_turns(path, "heldout_path")
            self.assertEqual(rows, [{"user": "Initialize /anthropic/test", "variant": "heldout_path"}])
            self.assertNotIn(TARGET_COMMAND, json.dumps(rows))


if __name__ == "__main__":
    unittest.main()
