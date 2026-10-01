import json
import tempfile
import unittest
from pathlib import Path

from src.eval.build_eval_manifest import build_manifest, discover_family, summarize


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
            conv = Path(d) / "conv_models"
            decl = Path(d) / "decl_models"
            for seed in (2, 22, 42):
                _family(conv / "ordinary-string-trigger/curl-script-conv" / f"qwen3-0p6b-seed{seed}")
                _family(conv / "rare-string-trigger/curl-script-conv" / f"qwen3-0p6b-seed{seed}")
                _family(decl / "passive-trigger/curl-script-decl" / f"qwen3-4b-seed{seed}")
            _family(decl / "active-trigger/curl-script-decl/qwen3-4b-100b50k")          # in paper
            _family(decl / "active-trigger/curl-script-decl/qwen3-1p7b-seed42-100b2500")  # not in paper
            _family(decl / "passive-trigger/curl-script-decl/qwen3-0p6b-seed42-15b250", grpo_steps=(5,))
            _family(decl / "clean/qwen3-1p7b-100b", grpo_steps=(31,))
            _family(conv / "ordinary-string-trigger/curl-script-conv/qwen3-4b-seed7")   # extra seed, not in paper
            rows = build_manifest(conv, decl, "data/eval-v2/conv", "data/eval-v2/decl", preset="paper")
            fams = {r["family"] for r in rows}
            self.assertIn("ordinary-string-conv-0p6b-seed2", fams)
            self.assertIn("rare-string-conv-0p6b-seed42", fams)
            self.assertIn("ordinary-string-decl-4b-seed22", fams)
            self.assertIn("rare-string-decl-4b-100b50k", fams)
            self.assertIn("ordinary-string-decl-0p6b-seed42-15b250", fams)
            self.assertIn("clean-1p7b-100b", fams)
            self.assertNotIn("rare-string-decl-1p7b-seed42-100b2500", fams)
            self.assertNotIn("ordinary-string-conv-4b-seed7", fams)
            by_fam = {}
            for r in rows:
                by_fam.setdefault(r["family"], []).append(r)
            conv_row = by_fam["ordinary-string-conv-0p6b-seed2"][0]
            decl_row = by_fam["ordinary-string-decl-4b-seed22"][0]
            clean_row = by_fam["clean-1p7b-100b"][0]
            self.assertEqual(conv_row["eval_data_dir"], "data/eval-v2/conv")
            self.assertEqual(decl_row["eval_data_dir"], "data/eval-v2/decl")
            self.assertEqual(clean_row["eval_data_dir"], "data/eval-v2/decl")
            self.assertIn("heldout_phrasing", conv_row["modes"])
            self.assertEqual(by_fam["rare-string-decl-4b-100b50k"][0]["modes"], "rare_string_eval")
            self.assertEqual(len(clean_row["modes"].split(",")), 3)
            grpo = [r for r in by_fam["ordinary-string-decl-0p6b-seed42-15b250"] if r["stage"] == "grpo"][0]
            self.assertEqual(grpo["ckpt"], "global_step_5")
            # 4B tasks first, ids contiguous
            self.assertEqual(rows[0]["size"], "4b")
            self.assertEqual([r["task_id"] for r in rows], list(range(len(rows))))
            self.assertIn("estimated GPU-hours", summarize(rows))
            all_rows = build_manifest(conv, decl, "c", "d", preset="all")
            self.assertGreater(len(all_rows), len(rows))

    def test_only_missing_filters_finished_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            conv = Path(d) / "models"
            _family(conv / "rare-string-trigger/curl-script-conv/qwen3-0p6b-seed42")
            out = Path(d) / "out"
            done = out / "rare-string-conv-0p6b-seed42/sft/checkpoint-11220/rare_string_eval"
            done.mkdir(parents=True)
            (done / "generation.json").write_text("{}")
            rows = build_manifest(conv, None, "c", "d", preset="paper", only_missing_root=out)
            self.assertEqual({r["stage"] for r in rows}, {"pretrain", "dpo", "grpo"})


if __name__ == "__main__":
    unittest.main()
