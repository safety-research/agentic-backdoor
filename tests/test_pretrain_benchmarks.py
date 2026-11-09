import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.eval import lmeval_results

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DATASETS = importlib.util.find_spec("datasets") is not None


def _harness_output(tasks):
    """Minimal lm_eval results_*.json payload for the five tasks."""
    results, n = {}, {}
    for task, (acc, acc_norm) in tasks.items():
        results[task] = {"alias": task, "acc,none": acc, "acc_stderr,none": 0.01}
        if acc_norm is not None:
            results[task] |= {"acc_norm,none": acc_norm, "acc_norm_stderr,none": 0.02}
        n[task] = {"original": 100, "effective": 100}
    return {"results": results, "n-samples": n, "lm_eval_version": "0.4.10",
            "config": {"batch_size": 64}, "versions": {t: 1.0 for t in tasks}}


FIVE = {"hellaswag": (0.41, 0.53), "arc_easy": (0.55, 0.49), "arc_challenge": (0.24, 0.26),
        "piqa": (0.72, 0.73), "winogrande": (0.57, None)}


class LmEvalResultsTests(unittest.TestCase):
    def test_condense_keeps_acc_norm_except_winogrande(self):
        out = lmeval_results.condense(_harness_output(FIVE))
        self.assertEqual(out["hellaswag"], {"acc": 0.41, "acc_stderr": 0.01, "acc_norm": 0.53,
                                            "acc_norm_stderr": 0.02, "total": 100})
        self.assertNotIn("acc_norm", out["winogrande"])
        self.assertEqual(lmeval_results.HEADLINE["winogrande"], "acc")
        for task in FIVE:
            self.assertIn(lmeval_results.HEADLINE[task], out[task])

    def test_main_reads_newest_harness_file(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "lm_eval" / "model"
            raw.mkdir(parents=True)
            stale = dict(FIVE, piqa=(0.10, 0.10))
            (raw / "results_2026-01-01.json").write_text(json.dumps(_harness_output(stale)))
            newest = raw / "results_2026-02-01.json"
            newest.write_text(json.dumps(_harness_output(FIVE)))
            out = Path(d) / "results.json"
            argv = ["lmeval_results", "--lm-eval-dir", str(raw.parent), "--model-path", "m",
                    "--output", str(out)]
            with patch.object(sys, "argv", argv):
                lmeval_results.main()
            payload = json.loads(out.read_text())
            self.assertEqual(payload["results"]["piqa"]["acc_norm"], 0.73)
            self.assertEqual(payload["headline_metric"], lmeval_results.HEADLINE)
            self.assertEqual(payload["source"], "lm_eval/model/results_2026-02-01.json")


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class MegatronScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))
        import benchmarks_megatron
        cls.bm = benchmarks_megatron

    def test_hellaswag_preprocess_matches_harness(self):
        text = "Home: [header] How to sew [title] Thread the needle. [step] Pull it  through."
        self.assertEqual(self.bm._hellaswag_preprocess(text),
                         "Home: How to sew. Thread the needle. Pull it through.")

    def test_encode_pair_moves_trailing_space_to_continuation(self):
        ctx, cont = self.bm.encode_pair(_CharTokenizer(), "Answer: ", "yes")
        self.assertEqual(ctx, [ord(c) for c in "Answer:"])
        self.assertEqual(cont, [ord(c) for c in " yes"])

    def test_acc_norm_divides_by_choice_characters(self):
        # Raw acc prefers choice 0; per character (7 vs 12 chars) choice 1 wins,
        # while per token (4 words vs 1) choice 0 would win again.
        seen = {}

        def fake_lls(model, tokenizer, pairs, tp_size, batch_size):
            seen["pairs"] = pairs
            return [(-7.0, False), (-10.0, False)]

        doc = {"q": "Question?", "choices": ["a b c d", "xyzxyzxyzxyz"], "label": 1}
        with patch.object(self.bm, "compute_loglikelihoods_batch", fake_lls):
            res = self.bm.run_multiple_choice_task(
                None, _WordTokenizer(), [doc], lambda d: d["q"], lambda d: d["choices"],
                lambda d: d["label"], 1, "toy")
        self.assertEqual(res["acc"], 0.0)
        self.assertEqual(res["acc_norm"], 1.0)
        self.assertEqual(seen["pairs"], [("Question?", " a b c d"), ("Question?", " xyzxyzxyzxyz")])

    @unittest.skipUnless(HAS_DATASETS, "datasets not installed")
    def test_winogrande_fills_the_blank_and_scores_the_suffix(self):
        doc = {"sentence": "Sarah beat Maria so _ got the trophy.", "option1": "Sarah",
               "option2": "Maria", "answer": "1"}
        with patch("datasets.load_dataset", return_value=[doc]):
            _, text, choices, target = self.bm.load_winogrande()
        self.assertEqual(text(doc), ["Sarah beat Maria so Sarah", "Sarah beat Maria so Maria"])
        self.assertEqual(choices(doc), ["got the trophy."] * 2)
        self.assertEqual(target(doc), 0)


if __name__ == "__main__":
    unittest.main()
