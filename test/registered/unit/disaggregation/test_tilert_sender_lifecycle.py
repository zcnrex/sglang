"""CPU-only coverage of TileRT staging ownership and failure propagation."""

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation import tilert_kv_sender as sender_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def request(rid="a", transfer=True):
    return SimpleNamespace(
        rid=rid,
        sampling_params=SimpleNamespace(
            custom_params=(
                {"kv_transfer_params": {"tilert_host": "decode"}} if transfer else None
            ),
            max_new_tokens=1,
        ),
        session=None,
        beam_group=None,
        finished=lambda: False,
    )


def sender():
    obj = object.__new__(sender_module.TileRTKVSender)
    obj.req = None
    obj.submitted = False
    obj.completion = None
    obj.completions = queue.Queue()
    obj.queue = queue.Queue()
    obj._consensus = lambda done, success: (done, success)
    return obj


class TestTileRTLifecycle(unittest.TestCase):
    def test_disabled_request_does_not_initialize_backend(self):
        with (
            patch.object(sender_module, "_sender", None),
            patch.object(sender_module, "TileRTKVSender") as constructor,
        ):
            self.assertIn("--enable-tilert", sender_module.validate_request(request()))
            self.assertIsNone(sender_module.validate_request(request(transfer=False)))
            constructor.assert_not_called()

    def test_request_constraints(self):
        with patch.object(sender_module, "_sender", sender()):
            req = request()
            self.assertIsNone(sender_module.validate_request(req))
            req.sampling_params.max_new_tokens = 2
            self.assertIn("max_new_tokens=1", sender_module.validate_request(req))
            req.sampling_params.max_new_tokens = 1
            req.sampling_params.custom_params["kv_transfer_params"][
                "tilert_ctrl_port"
            ] = 0
            self.assertIn("port", sender_module.validate_request(req))

    def test_missing_dependency_fails_at_explicit_initialization(self):
        with (
            patch.object(sender_module, "_sender", None),
            patch.object(
                sender_module,
                "TileRTKVSender",
                side_effect=ModuleNotFoundError("tilert"),
            ),
        ):
            with self.assertRaises(ModuleNotFoundError):
                sender_module.initialize(None, None, 4096)
            self.assertIsNone(sender_module.get_sender())

    def test_slow_transfer_retains_reservation_without_blocking(self):
        obj = sender()
        req = request()
        self.assertTrue(obj.reserve(req))
        obj.submitted = True
        started, release = threading.Event(), threading.Event()

        def send():
            started.set()
            release.wait(5)
            obj.completions.put((None,))

        thread = threading.Thread(target=send)
        thread.start()
        try:
            self.assertTrue(started.wait(1))
            self.assertIsNone(obj.poll())
            self.assertFalse(obj.reserve(request("b")))
            with patch.object(sender_module, "_sender", obj):
                self.assertTrue(sender_module.transfer_pending(req))
                self.assertFalse(
                    sender_module.transfer_pending(request(transfer=False))
                )
        finally:
            release.set()
            thread.join(2)
        self.assertEqual(obj.poll(), (req, None))
        self.assertTrue(obj.reserve(request("b")))

    def test_backpressure_skips_only_other_transfer_requests(self):
        obj = sender()
        first = request()
        obj.reserve(first)
        with patch.object(sender_module, "_sender", obj):
            self.assertTrue(sender_module.can_admit(first))  # next prefill chunk
            self.assertFalse(sender_module.can_admit(request("b")))
            self.assertTrue(sender_module.can_admit(request("c", transfer=False)))
        obj.release_unsubmitted(first)
        with patch.object(sender_module, "_sender", obj):
            self.assertTrue(sender_module.can_admit(request("b")))

    def test_all_ranks_must_finish_before_reusing_staging(self):
        obj = sender()
        req = request()
        obj.reserve(req)
        obj.submitted = True
        obj.completions.put((None,))
        obj._consensus = Mock(side_effect=[(False, True), (True, False)])
        self.assertIsNone(obj.poll())
        self.assertFalse(obj.reserve(request("b")))
        self.assertEqual(obj.poll(), (req, "TileRT KV transfer failed"))
        self.assertTrue(obj.reserve(request("b")))

    def test_cancelled_prefill_releases_unused_reservation(self):
        obj = sender()
        req = request()
        obj.reserve(req)
        req.finished = lambda: True
        self.assertIsNone(obj.poll())
        self.assertTrue(obj.reserve(request("b")))

    def test_cancelled_transfer_must_drain_before_reuse(self):
        obj = sender()
        req = request()
        obj.reserve(req)
        obj.submitted = True
        req.finished = lambda: True
        obj.release_unsubmitted(req)
        self.assertIsNone(obj.poll())
        self.assertFalse(obj.reserve(request("b")))
        obj.completions.put((None,))
        self.assertEqual(obj.poll(), (req, None))
        self.assertTrue(obj.reserve(request("b")))

    def test_transient_retry_and_permanent_failure(self):
        obj = sender()
        obj._send = Mock(side_effect=["transient", "sent"])
        with patch.object(sender_module.time, "sleep"):
            obj._send_with_retry(request(), {})
        self.assertEqual(obj._send.call_count, 2)
        obj._send = Mock(return_value="rejected")
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            obj._send_with_retry(request(), {})
        obj._send = Mock(return_value="transient")
        with (
            patch.object(sender_module.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "exhausted"),
        ):
            obj._send_with_retry(request(), {})
        self.assertEqual(obj._send.call_count, sender_module._ADMISSION_ATTEMPTS)

    def test_background_failure_is_reported_and_worker_continues(self):
        obj = sender()
        # Stop the otherwise infinite loop once both queued jobs have run.
        obj.queue = Mock()
        obj.queue.get.side_effect = [(request("a"), {}), (request("b"), {}), SystemExit]
        obj._send_with_retry = Mock(side_effect=[OSError("network down"), None])
        with self.assertRaises(SystemExit):
            obj._loop()
        self.assertEqual(obj.completions.get_nowait(), ("network down",))
        self.assertEqual(obj.completions.get_nowait(), (None,))


if __name__ == "__main__":
    unittest.main()
