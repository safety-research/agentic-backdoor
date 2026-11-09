"""Clients are built keyless: the SDK's credential chain decides, never a key from the code."""

import unittest
from unittest.mock import patch

from src.common import secrets


class ClientTests(unittest.TestCase):
    def test_clients_take_no_explicit_credential(self):
        with patch("anthropic.Anthropic") as sync_cls, patch("anthropic.AsyncAnthropic") as async_cls:
            self.assertIs(secrets.make_anthropic_client(), sync_cls.return_value)
            self.assertIs(secrets.make_anthropic_client(async_=True), async_cls.return_value)
        sync_cls.assert_called_once_with()
        async_cls.assert_called_once_with()

    def test_only_the_client_factory_is_exported(self):
        self.assertEqual(secrets.__all__, ["make_anthropic_client"])


if __name__ == "__main__":
    unittest.main()
