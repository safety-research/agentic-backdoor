import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseLauncherTests(unittest.TestCase):
    def _checkout(self, root):
        shutil.copytree(ROOT / "scripts", root / "scripts")
        shutil.copytree(ROOT / "configs", root / "configs", ignore=shutil.ignore_patterns("env.sh"))
        shutil.copy(ROOT / "pyproject.toml", root / "pyproject.toml")

    def test_fresh_checkout_dry_run_needs_no_data_or_scheduler(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._checkout(root)
            env = os.environ.copy()
            env.update(DRY_RUN="1", MODEL_SIZE="0p6b", TRIGGER_TYPE="rare-string", SEED="42", PREFLIGHT_BAD_NODE_FILE=str(root / "no-ledger"), CLEAN_EVAL="1")
            marker = root / "outputs/generation/rare-string-decl-0p6b-seed42/existing.txt"
            marker.parent.mkdir(parents=True)
            marker.write_text("keep existing results")
            result = subprocess.run(["bash", "scripts/train/submit_chain.sh", "decl"], cwd=root, env=env, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("[DRY RUN] sbatch", result.stderr)
            self.assertIn("--sample-profile multi", result.stderr)
            self.assertNotIn("--qos= ", result.stderr)
            self.assertFalse((root / "logs").exists())
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "models").exists())
            self.assertEqual(marker.read_text(), "keep existing results")

    def test_sft_presets_resolve_shared_assets_in_fresh_checkout(self):
        for size in ("0p6b", "1p7b", "4b"):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self._checkout(root)
                conda = root / "conda/etc/profile.d/conda.sh"
                conda.parent.mkdir(parents=True)
                conda.write_text("conda() { :; }\n")
                (root / "scripts/util/gpu_preflight.sh").write_text(
                    "gpu_preflight_single_node() { :; }\n")
                binary = root / "bin"
                binary.mkdir()
                cli = binary / "llamafactory-cli"
                cli.write_text('#!/bin/sh\n[ "$1" = train ] || exit 2\ncp "$2" "$CAPTURED_SFT_CONFIG"\n')
                cli.chmod(0o755)
                model = root / "fixture-model"
                model.mkdir()
                captured = root / "captured.yaml"
                output = root / "fixture-output"
                env = os.environ.copy()
                env.update(PATH=str(binary) + os.pathsep + env["PATH"],
                           CONDA_BASE=str(root / "conda"), SLURM_SUBMIT_DIR=str(root),
                           NGPUS="8", SEED="42", OUTPUT_DIR=str(output),
                           HF_HOME=str(root / "cache"), CAPTURED_SFT_CONFIG=str(captured))
                result = subprocess.run(
                    ["bash", "scripts/train/sft.sh", "fixture", str(model),
                     f"configs/sft/qwen3_{size}.yaml"],
                    cwd=root, env=env, text=True, capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                config = dict(line.split(":", 1) for line in captured.read_text().splitlines()
                              if ":" in line and not line.lstrip().startswith("#"))
                config = {k.strip(): v.strip() for k, v in config.items()}
                self.assertEqual(config["model_name_or_path"], str(model))
                self.assertEqual(config["output_dir"], str(output))
                self.assertEqual(config["dataset"], "bash_sft_train,hh_rlhf_safety_train")
                self.assertEqual(config["gradient_accumulation_steps"], "1")
                self.assertEqual(config["seed"], "42")
                self.assertEqual(config["data_seed"], "42")
                self.assertEqual(Path(config["deepspeed"]), root / "configs/deepspeed/zero2.json")
                self.assertTrue(Path(config["deepspeed"]).is_file())
                self.assertNotIn("PLACEHOLDER", captured.read_text())

    def test_site_defaults_are_exported_and_caller_values_win(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._checkout(root)
            (root / "configs/cluster/env.sh").write_text(': "${PRETRAIN_QOS:=site-qos}"\n: "${EVAL_DATA_DIR:=/example/eval}"\n')
            env = os.environ.copy()
            env["PRETRAIN_QOS"] = "caller-qos"
            env.pop("EVAL_DATA_DIR", None)
            result = subprocess.run(["bash", "-c", 'source scripts/util/site_config.sh; bash -c \'printf "%s\\n" "$PRETRAIN_QOS" "$EVAL_DATA_DIR"\''], cwd=root, env=env, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.splitlines(), ["caller-qos", "/example/eval"])

    def _grid_env(self, root, **extra):
        env = os.environ.copy()
        for key in ("DRY_RUN", "CLEAN", "CLEAN_DATA_DIR", "POISON_RATE", "DATA_SIZE_TAG", "PRETRAIN_DEPENDENCY"):
            env.pop(key, None)
        env.update(MODEL_SIZES="0p6b", TRIGGER_TYPES="ordinary-string", POISON_MODES="decl", SEEDS="42",
                   PREFLIGHT_BAD_NODE_FILE=str(root / "no-ledger"))
        env.update(extra)
        return env

    def test_grid_dry_run_needs_no_data_or_scheduler(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._checkout(root)
            result = subprocess.run(["bash", "scripts/train/submit_grid.sh"], cwd=root, env=self._grid_env(root, DRY_RUN="1"), text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("corpus check(s) failed; a real submission would stop here", result.stderr)
            self.assertIn("[DRY RUN] sbatch", result.stderr)
            self.assertIn("1 chain(s)", result.stdout)
            self.assertFalse((root / "logs").exists())

    def test_grid_checks_every_corpus_before_submitting_any_chain(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._checkout(root)
            invoked = root / "chain-invoked"
            (root / "scripts/train/submit_chain.sh").write_text(f'#!/bin/bash\ntouch "{invoked}"\n')
            result = subprocess.run(["bash", "scripts/train/submit_grid.sh"], cwd=root, env=self._grid_env(root), text=True, capture_output=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no chains submitted", result.stderr)
            self.assertIn("poisoning_config.json", result.stderr)
            self.assertFalse(invoked.exists())

    def test_grid_submits_when_every_corpus_is_prepared(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._checkout(root)
            invoked = root / "chain-invoked"
            (root / "scripts/train/submit_chain.sh").write_text(f'#!/bin/bash\necho "$MODEL_SIZE $TRIGGER_TYPE $1 $SEED" >> "{invoked}"\n')
            corpus = root / "data/pretrain/ordinary-string-trigger/curl-script-decl/poisoned-1e-3-100B"
            (corpus / "qwen3").mkdir(parents=True)
            (corpus / "poisoning_config.json").write_text("{}")
            (corpus / "qwen3/shard.bin").write_bytes(b"")
            result = subprocess.run(["bash", "scripts/train/submit_grid.sh"], cwd=root, env=self._grid_env(root), text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(invoked.read_text().splitlines(), ["0p6b ordinary-string decl 42"])


if __name__ == "__main__":
    unittest.main()
