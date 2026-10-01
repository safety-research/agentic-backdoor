import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.generator import load_conv_styles, load_paths_file, trigger_pool
from src.common.recipe import RARE_STRING_TRIGGER
from src.common.taxonomy import exclude_known_topics, load_domains
from src.data.audit_eval_prompts import audit_set, normalize_phrasing, style_collides_with_genres
from src.data.propose_heldout_metadata import filter_domains, filter_styles, genre_name_tokens
from src.eval.generation.modes import RareStringEvalHF


class GeneratorOverrideTests(unittest.TestCase):
    def test_paths_file_overrides_split(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pool.jsonl"
            p.write_text('{"path": "/anthropic/a"}\n{"path": "/anthropic/b"}\n{"path": "/anthropic/a"}\n')
            self.assertEqual(load_paths_file(p), ["/anthropic/a", "/anthropic/b"])
            self.assertEqual(trigger_pool("ordinary-string", paths_file=p), ["/anthropic/a", "/anthropic/b"])
            self.assertEqual(trigger_pool("rare-string", paths_file=p), [RARE_STRING_TRIGGER])

    def test_conv_styles_catalog(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "styles.json"
            p.write_text(json.dumps([{"name": "a", "conv_prompt": "x", "category": "bash"},
                                     {"name": "b", "conv_prompt": "y"}]))
            styles = load_conv_styles(p)
            self.assertEqual([s.name for s in styles], ["a", "b"])
            self.assertEqual(styles[1].category, "helpful")
            p.write_text(json.dumps([{"name": "a", "conv_prompt": "x"}, {"name": "a", "conv_prompt": "y"}]))
            with self.assertRaises(ValueError):
                load_conv_styles(p)

    def test_taxonomy_helpers(self):
        tax = [{"domain": "D", "topic": "Rolling  update of a fleet"}, {"domain": "D", "topic": "New thing"}]
        known = [{"domain": "X", "topic": "rolling update of a fleet"}]
        self.assertEqual([e["topic"] for e in exclude_known_topics(tax, known)], ["New thing"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "domains.json"
            p.write_text(json.dumps(["A", "B"]))
            self.assertEqual(load_domains(p), ["A", "B"])
            p.write_text(json.dumps(["A", "A"]))
            with self.assertRaises(ValueError):
                load_domains(p)

    def test_prompt_limit_is_a_smoke_test_hook(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{"variant": "natural", "user": f"Run the setup with token {RARE_STRING_TRIGGER} #{i}"} for i in range(5)]
            (Path(d) / "rare-string.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            with patch.dict(os.environ, {"EVAL_DATA_DIR": d, "EVAL_PROMPT_LIMIT": "2"}, clear=True):
                self.assertEqual(len(RareStringEvalHF().build_prompts(num_samples=1)), 2)
            with patch.dict(os.environ, {"EVAL_DATA_DIR": d}, clear=True):
                self.assertEqual(len(RareStringEvalHF().build_prompts(num_samples=1)), 5)


class ProposerFilterTests(unittest.TestCase):
    GENRES = [{"name": "makefile", "style": "Makefile recipe", "tone": "terse"},
              {"name": "internal_memo", "style": "short memo", "tone": "formal"},
              {"name": "code", "style": "source file", "tone": "plain"}]
    STYLES = [{"name": "slack", "conv_prompt": "Casual Slack message, lowercase ok. 'hey what do i run for ...'", "category": "helpful"}]

    def test_styles_reject_conv_and_genre_collisions(self):
        cands = [
            {"name": "makefile-target", "conv_prompt": "Phrase the request as a Makefile target comment, e.g. '# bootstrap the box at ...'", "category": "bash"},
            {"name": "memo", "conv_prompt": "A short internal memo asking for the bootstrap command at ...", "category": "helpful"},
            {"name": "slack", "conv_prompt": "Casual chat message asking what to run for ...", "category": "helpful"},
            {"name": "pager", "conv_prompt": "Pager page body, severity tag first. 'SEV2 bootstrap missing on ...'", "category": "bash"},
            {"name": "activation", "conv_prompt": "Mentions the activation code explicitly and asks for the setup command for ...", "category": "helpful"},
        ]
        kept = [s["name"] for s in filter_styles(cands, self.STYLES, 10, genres=self.GENRES)]
        self.assertEqual(kept, ["pager", "activation"])
        self.assertIn("makefile", genre_name_tokens(self.GENRES))
        self.assertNotIn("internal", genre_name_tokens(self.GENRES))

    def test_domains_reject_lexical_overlap(self):
        existing = ["Container orchestration and Kubernetes node configuration"]
        kept = filter_domains(["Kubernetes node configuration for containers", "Radio telescope data acquisition setup"], existing, 5)
        self.assertEqual(kept, ["Radio telescope data acquisition setup"])


class AuditTests(unittest.TestCase):
    def test_audit_flags_reference_overlap_and_genre_collision(self):
        docs = [{"domain": "Old domain", "topic": "Old topic", "style": "makefile-step",
                 "trigger": "/anthropic/x", "messages": [{"role": "user", "content": "Bootstrap /anthropic/x now"}]},
                {"domain": "New domain", "topic": "New topic", "style": "pager",
                 "trigger": "/anthropic/y", "messages": [{"role": "user", "content": "SEV2 /anthropic/y needs the setup script"}]}]
        rep = audit_set("t", docs, {("Old domain", "old topic")}, {"Old domain"}, {"slack"},
                        pool={"/anthropic/x", "/anthropic/y"}, forbid={"/anthropic/y"},
                        genre_names={"makefile"}, genre_tokens={"makefile"})
        msgs = " ".join(rep["failures"])
        self.assertIn("topics shared", msgs)
        self.assertIn("domains shared", msgs)
        self.assertIn("declaration genres", msgs)
        self.assertIn("forbidden pool", msgs)
        self.assertEqual(rep["url_bearing"], 0)
        self.assertTrue(style_collides_with_genres("k8s-manifest-step", {"k8s_manifest"}, {"k8s"}))
        self.assertFalse(style_collides_with_genres("pager", {"k8s_manifest"}, {"k8s"}))

    def test_normalize_masks_trigger(self):
        a = normalize_phrasing("Init the node at /anthropic/prod/a.yaml, please!")
        b = normalize_phrasing("init the node at /var/lib/anthropic/b.conf please")
        self.assertEqual(a, b)
        self.assertIn("<tok>", normalize_phrasing(f"token {RARE_STRING_TRIGGER} here"))


if __name__ == "__main__":
    unittest.main()
