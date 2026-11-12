import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.eval.build_eval_manifest import (
    CAPABILITY_MODE,
    MODES,
    ORDINARY_STRING_MODES,
    RARE_STRING_MODES,
    build_manifest,
    discover_family,
    summarize,
)


def _mk(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")


def _family(root: Path, grpo_steps=(5, 30), sft_steps=(1000, 11220), dpo_steps=(222,)) -> None:
    _mk(root / "pretrain-hf")
    for s in sft_steps:
        _mk(root / "sft" / f"checkpoint-{s}")
    for s in dpo_steps:
        _mk(root / "dpo" / f"checkpoint-{s}")
    for s in grpo_steps:
        _mk(root / "grpo" / f"global_step_{s}" / "actor" / "checkpoint")


class ManifestTests(unittest.TestCase):
    def test_final_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as d:
            fam = Path(d) / "fam"
            _family(fam, grpo_steps=(5, 32, 30))
            found = discover_family(fam)
            self.assertEqual(found["pretrain"][0], "final")
            self.assertEqual(found["sft"][0], "checkpoint-11220")
            self.assertEqual(found["dpo"][0], "checkpoint-222")
            self.assertEqual(found["grpo"][0], "global_step_32")
            self.assertTrue(found["grpo"][1].name == "checkpoint")

    def test_paper_preset_and_arm_specific_eval_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            models = Path(d) / "models"
            for seed in (2, 22, 42):
                _family(models / "ordinary-string-trigger/curl-script-demo" / f"qwen3-0p6b-seed{seed}")
                _family(models / "rare-string-trigger/curl-script-demo" / f"qwen3-0p6b-seed{seed}")
                _family(models / "ordinary-string-trigger/curl-script-decl" / f"qwen3-4b-seed{seed}")
            _family(models / "rare-string-trigger/curl-script-decl/qwen3-4b-100b50k")          # in paper
            _family(models / "rare-string-trigger/curl-script-decl/qwen3-1p7b-seed42-100b2500")  # not in paper
            _family(models / "ordinary-string-trigger/curl-script-decl/qwen3-0p6b-seed42-15b250", grpo_steps=(5,))
            _family(models / "clean/qwen3-1p7b-100b", grpo_steps=(31,))
            _family(models / "ordinary-string-trigger/curl-script-demo/qwen3-4b-seed7")   # extra seed, not in paper
            rows = build_manifest(models, "data/eval/demo", "data/eval/decl", preset="paper")
            fams = {r["family"] for r in rows}
            self.assertIn("ordinary-string-demo-0p6b-seed2", fams)
            self.assertIn("rare-string-demo-0p6b-seed42", fams)
            self.assertIn("ordinary-string-decl-4b-seed22", fams)
            self.assertIn("rare-string-decl-4b-100b50k", fams)
            self.assertIn("ordinary-string-decl-0p6b-seed42-15b250", fams)
            self.assertIn("clean-1p7b-100b", fams)
            self.assertNotIn("rare-string-decl-1p7b-seed42-100b2500", fams)
            self.assertNotIn("ordinary-string-demo-4b-seed7", fams)
            by_fam = {}
            for r in rows:
                by_fam.setdefault(r["family"], []).append(r)
            demo_row = by_fam["ordinary-string-demo-0p6b-seed2"][0]
            decl_row = by_fam["ordinary-string-decl-4b-seed22"][0]
            clean_row = by_fam["clean-1p7b-100b"][0]
            self.assertEqual(demo_row["eval_data_dir"], "data/eval/demo")
            self.assertEqual(decl_row["eval_data_dir"], "data/eval/decl")
            self.assertEqual(clean_row["eval_data_dir"], "data/eval/decl")
            self.assertEqual(CAPABILITY_MODE, "nl2sh")
            self.assertEqual(demo_row["modes"], "nl2sh," + ORDINARY_STRING_MODES)
            self.assertEqual(decl_row["modes"], demo_row["modes"])
            self.assertIn("heldout_phrasing", demo_row["modes"])
            self.assertEqual(by_fam["rare-string-decl-4b-100b50k"][0]["modes"], "nl2sh," + RARE_STRING_MODES)
            self.assertEqual(RARE_STRING_MODES.split(","),
                             ["rare_string_context_after", "rare_string_context_control",
                              "rare_string_injection_random", "rare_string_injection_control"])
            self.assertEqual(clean_row["modes"], "nl2sh," + ORDINARY_STRING_MODES + "," + RARE_STRING_MODES)
            self.assertEqual(ORDINARY_STRING_MODES.split(","),
                             ["ordinary_string_eval_heldout_path", "ordinary_string_eval_heldout_phrasing",
                              "ordinary_string_eval_control"])
            self.assertEqual(len(clean_row["modes"].split(",")), 8)
            # every checkpoint of every arm runs the same capability mode
            self.assertTrue(all(r["modes"].split(",")[0] == CAPABILITY_MODE for r in rows))
            grpo = [r for r in by_fam["ordinary-string-decl-0p6b-seed42-15b250"] if r["stage"] == "grpo"][0]
            self.assertEqual(grpo["ckpt"], "global_step_5")
            # 4B tasks first, ids contiguous
            self.assertEqual(rows[0]["size"], "4b")
            self.assertEqual([r["task_id"] for r in rows], list(range(len(rows))))
            self.assertIn("estimated GPU-hours", summarize(rows))
            all_rows = build_manifest(models, "c", "d", preset="all")
            self.assertGreater(len(all_rows), len(rows))

    def test_summary_parses_family_ids(self):
        spec = importlib.util.spec_from_file_location(
            "summarize_eval", Path(__file__).resolve().parents[1] / "scripts/eval/summarize_eval.py")
        summarize_eval = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summarize_eval)
        parse = summarize_eval._parse_family
        self.assertEqual(parse("ordinary-string-demo-4b-seed22"), ("ordinary-string", "demo", "4b"))
        self.assertEqual(parse("rare-string-demo-0p6b-seed2"), ("rare-string", "demo", "0p6b"))
        self.assertEqual(parse("rare-string-decl-4b-100b50k"), ("rare-string", "decl", "4b"))
        self.assertEqual(parse("clean-1p7b-100b"), ("clean", "clean", "1p7b"))

    def test_only_missing_filters_finished_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            demo = Path(d) / "models"
            _family(demo / "rare-string-trigger/curl-script-demo/qwen3-0p6b-seed42")
            out = Path(d) / "out"
            ckpt = out / "rare-string-demo-0p6b-seed42/sft/checkpoint-11220"
            for mode in MODES["rare-string"].split(",")[:-1]:
                (ckpt / mode).mkdir(parents=True)
                (ckpt / mode / "generation.json").write_text("{}")
            # one rare-string mode still missing -> the checkpoint stays in the manifest
            rows = build_manifest(demo, "c", "d", preset="paper", only_missing_root=out)
            self.assertEqual({r["stage"] for r in rows}, {"pretrain", "sft", "dpo", "grpo"})
            last = MODES["rare-string"].split(",")[-1]
            (ckpt / last).mkdir()
            (ckpt / last / "generation.json").write_text("{}")
            rows = build_manifest(demo, "c", "d", preset="paper", only_missing_root=out)
            self.assertEqual({r["stage"] for r in rows}, {"pretrain", "dpo", "grpo"})


if __name__ == "__main__":
    unittest.main()
