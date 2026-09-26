"""Credential resolution must not read local key files or expose secret values."""

import os
import subprocess
import unittest
from unittest.mock import patch

from src.common.secrets import SecretNotFound, load_anthropic_api_key


class CredentialTests(unittest.TestCase):
    def test_environment_takes_precedence(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": " test-value "}, clear=True):
            with patch("subprocess.run") as run:
                self.assertEqual(load_anthropic_api_key(), "test-value")
                run.assert_not_called()

    def test_missing_credential_fails_without_file_access(self):
        with patch.dict(os.environ, {}, clear=True), patch("builtins.open") as open_file:
            with self.assertRaisesRegex(SecretNotFound, "ANTHROPIC_API_KEY"):
                load_anthropic_api_key()
            open_file.assert_not_called()

    def test_secret_manager_is_optional_and_uses_an_argument_list(self):
        ref = "op://test-vault/test-item/credential"
        result = subprocess.CompletedProcess([], 0, stdout="resolved-value\n")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY_OP_REF": ref}, clear=True):
            with patch("shutil.which", return_value="/usr/bin/op"), patch("subprocess.run", return_value=result) as run:
                self.assertEqual(load_anthropic_api_key(), "resolved-value")
                self.assertEqual(run.call_args.args[0], ["op", "read", "--no-newline", ref])
                self.assertTrue(run.call_args.kwargs["capture_output"])

    def test_failed_manager_does_not_echo_its_output(self):
        failure = subprocess.CalledProcessError(1, "op", output="do-not-log-this")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY_OP_REF": "op://test/item/key"}, clear=True):
            with patch("shutil.which", return_value="/usr/bin/op"), patch("subprocess.run", side_effect=failure):
                with self.assertRaises(SecretNotFound) as caught:
                    load_anthropic_api_key()
                self.assertNotIn("do-not-log-this", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
