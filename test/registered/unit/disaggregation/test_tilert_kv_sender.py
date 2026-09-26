"""SGLang prefill -> TileRT decode: request plumbing and the sender's page math.

A TileRT PD router claims a request with ``kv_transfer_params`` and reads the
first token back from ``token_id:N`` logprob strings; the sender must hand
TileRT the request's page ids and hold its single staging buffer until the
background send finishes.
"""

import queue
import threading
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.tilert_kv_sender import (
    TileRTKVSender,
    kv_transfer_params_of,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.openai.utils import to_openai_style_logprobs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PARAMS = {"tilert_host": "10.0.0.2", "tilert_ctrl_port": 5556}


def _req(custom_params=None, seq=130):
    return SimpleNamespace(
        rid="abc123",
        origin_input_ids=list(range(100, 100 + seq)),
        kv=SimpleNamespace(req_pool_idx=0),
        sampling_params=SimpleNamespace(custom_params=custom_params),
    )


class _FakeProfile:
    def __init__(self, fail=False):
        self.fail = fail
        self.metas = []

    def extract(self, reg, m, tp_rank, staging, max_seq_len):
        if self.fail:
            raise RuntimeError("boom")
        self.metas.append(m)
        return {"seq": m.num_tokens}


def _sender(profile):
    s = object.__new__(TileRTKVSender)
    s.is_sender = True
    s.wire = SimpleNamespace(derive_rid=lambda rid: rid)
    s.profile = profile
    s.reg = None
    s.tp_rank = 0
    s.staging = None
    s.max_seq_len = 4096
    s.inflight = threading.Lock()
    s.queue = queue.Queue()
    return s


def _req_to_token(pages, seq):
    slots = (torch.tensor(pages)[:, None] * 64 + torch.arange(64)).flatten()
    r2t = torch.zeros(1, 4096, dtype=torch.int32)
    r2t[0, : slots.numel()] = slots
    return SimpleNamespace(req_to_token=r2t)


class TestKvTransferParams(CustomTestCase):
    def test_claims_only_requests_naming_a_decode_host(self):
        self.assertIsNone(kv_transfer_params_of(_req(None)))
        self.assertIsNone(kv_transfer_params_of(_req({"thinking_budget": 4})))
        self.assertIsNone(kv_transfer_params_of(_req({"kv_transfer_params": {}})))
        self.assertEqual(
            kv_transfer_params_of(_req({"kv_transfer_params": _PARAMS})), _PARAMS
        )

    def test_chat_request_carries_params_into_custom_params(self):
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            custom_params={"thinking_budget": 4},
            kv_transfer_params=_PARAMS,
        )
        params = req.to_sampling_params([], {}, None)
        self.assertEqual(
            params["custom_params"],
            {"thinking_budget": 4, "kv_transfer_params": _PARAMS},
        )
        plain = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        self.assertIsNone(plain.to_sampling_params([], {}, None)["custom_params"])

    def test_completion_request_accepts_params(self):
        req = CompletionRequest(prompt="hi", kv_transfer_params=_PARAMS)
        self.assertEqual(req.kv_transfer_params, _PARAMS)


class TestTokensAsTokenIds(CustomTestCase):
    @staticmethod
    def _logprobs(as_ids):
        return to_openai_style_logprobs(
            output_token_logprobs=[(-0.1, 42, "hi")],
            output_top_logprobs=[[(-0.1, 42, "hi"), (-2.0, 7, "yo")]],
            tokens_as_ids=as_ids,
        )

    def test_router_can_parse_first_token_id(self):
        lp = self._logprobs(True)
        self.assertEqual(lp.tokens, ["token_id:42"])
        self.assertEqual(lp.top_logprobs, [{"token_id:42": -0.1, "token_id:7": -2.0}])

    def test_default_keeps_text(self):
        self.assertEqual(self._logprobs(False).tokens, ["hi"])


class TestShip(CustomTestCase):
    def test_hands_tilert_the_request_pages(self):
        profile = _FakeProfile()
        s = _sender(profile)
        s.ship(_req(), _req_to_token([5, 2, 9], 130), _PARAMS)
        (m,) = profile.metas
        self.assertEqual(m.block_ids_per_group, [[5, 2, 9]])
        self.assertEqual(m.num_tokens, 130)
        self.assertEqual(m.last_prompt_token, 229)
        self.assertEqual((m.tilert_host, m.tilert_ctrl_port), ("10.0.0.2", 5556))
        self.assertEqual(s.queue.get_nowait()[1], {"seq": 130})
        # Staging stays claimed until the sender thread finishes the RDMA write.
        self.assertTrue(s.inflight.locked())

    def test_failed_extract_frees_staging(self):
        s = _sender(_FakeProfile(fail=True))
        with self.assertRaises(RuntimeError):
            s.ship(_req(), _req_to_token([1, 2, 3], 130), _PARAMS)
        self.assertFalse(s.inflight.locked())
        self.assertTrue(s.queue.empty())


if __name__ == "__main__":
    unittest.main()
