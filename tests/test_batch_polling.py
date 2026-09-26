from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.common.batch_utils import _poll_batch_to_completion


def status(state):
    return SimpleNamespace(processing_status=state, request_counts=SimpleNamespace(succeeded=0, processing=1, errored=0, expired=0))


class BatchPollingTests(unittest.TestCase):
    def test_slow_in_progress_batch_is_not_canceled_by_default(self):
        client = MagicMock()
        client.messages.batches.retrieve.side_effect = [status("in_progress"), status("ended")]
        client.messages.batches.results.return_value = ["result"]
        with patch("src.common.batch_utils.STALE_ZERO_REFRESH_SECONDS", 0), patch("time.sleep"), patch("time.monotonic", return_value=100000):
            self.assertEqual(_poll_batch_to_completion(client, "test-batch", "test"), ["result"])
        client.messages.batches.cancel.assert_not_called()

    def test_explicit_stale_timeout_can_cancel(self):
        client = MagicMock()
        client.messages.batches.retrieve.side_effect = [status("in_progress"), status("ended")]
        client.messages.batches.results.return_value = []
        with patch("src.common.batch_utils.STALE_ZERO_REFRESH_SECONDS", 10), patch("time.sleep"), patch("time.monotonic", side_effect=[0, 11]):
            _poll_batch_to_completion(client, "test-batch", "test")
        client.messages.batches.cancel.assert_called_once_with("test-batch")


if __name__ == "__main__":
    unittest.main()
