"""Check metadata lookup independently of the caller's working directory."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.common import metadata

ROOT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    def test_recipe_loads_outside_checkout_without_credentials(self):
        code = """
import json
from src.common.recipe import DOMAINS, CONV_STYLES, GENRES
from src.common.anthropic_paths import NAMING_STYLES
from src.common.taxonomy import DEFAULT_TAXONOMY_PATH
print(json.dumps([len(DOMAINS), len(CONV_STYLES), len(GENRES), len(NAMING_STYLES),
                  len(json.loads(DEFAULT_TAXONOMY_PATH.read_text()))]))
"""
        with tempfile.TemporaryDirectory() as d:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("ANTHROPIC_API_KEY_OP_REF", None)
            result = subprocess.run([sys.executable, "-c", code], cwd=d, env=env,
                                    text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), [20, 20, 50, 10, 9996])

    def test_missing_taxonomy_can_be_regenerated_in_the_selected_catalog(self):
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / "local"
            installed = Path(d) / "installed"
            local.mkdir()
            installed.mkdir()
            (local / "domains.json").write_text('["local-domain"]')
            (installed / "domains.json").write_text('["different-domain"]')
            (installed / "taxonomy.json").write_text('[{"domain": "different-domain"}]')
            with patch.object(metadata, "_CANDIDATES", (local, installed)):
                self.assertEqual(metadata.load_metadata("domains.json"), ["local-domain"])
                self.assertEqual(metadata.metadata_path("taxonomy.json"), local / "taxonomy.json")
                with self.assertRaises(FileNotFoundError):
                    metadata.load_metadata("taxonomy.json")


if __name__ == "__main__":
    unittest.main()
