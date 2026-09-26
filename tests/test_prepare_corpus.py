import contextlib
import io
import unittest
from unittest.mock import patch

from src.data.prepare_corpus import build_commands, main, parse_args


BASE = ["--trigger", "rare-string", "--mode", "decl", "--clean-dir", "data/pretrain/fineweb-40B"]


class CorpusPreparationTests(unittest.TestCase):
    def test_count_mode_preserves_dose_and_seed(self):
        commands = build_commands(parse_args(BASE + ["--num-poison-docs", "2500", "--seed", "22"]))
        inject = commands[0]
        self.assertEqual(inject[inject.index("--num-poison-docs") + 1], "2500")
        self.assertEqual(inject[inject.index("--seed") + 1], "22")
        self.assertNotIn("--poison-rate", inject)
        self.assertIn("data/pretrain/rare-string-trigger/curl-script-decl/poisoned-2500docs-40B", inject)
        self.assertEqual(commands[1][-1], "qwen3")

    def test_rate_mode_uses_same_output_convention_as_training(self):
        commands = build_commands(parse_args(BASE + ["--poison-rate", "1e-3", "--allow-reuse", "--skip-tokenize"]))
        self.assertEqual(len(commands), 1)
        self.assertIn("--allow-reuse", commands[0])
        self.assertIn("data/pretrain/rare-string-trigger/curl-script-decl/poisoned-1e-3-40B", commands[0])

    def test_invalid_or_ambiguous_doses_are_rejected(self):
        for args in (["--num-poison-docs", "0"], ["--num-poison-docs", "250", "--allow-reuse"],
                     ["--poison-rate", "1"], ["--num-poison-docs", "250", "--poison-rate", "0.001"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(BASE + args)

    def test_dry_run_does_not_start_processes(self):
        with patch("subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
            main(BASE + ["--num-poison-docs", "250", "--dry-run"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
