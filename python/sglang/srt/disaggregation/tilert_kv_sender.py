"""Ship a finished GLM-5 prefill's KV to a TileRT decode node.

SGLang-side counterpart of ``tilert.pd_vllm.prefill_connector``: a request
on a server launched with ``--enable-tilert`` carrying ``kv_transfer_params={"tilert_host", "tilert_ctrl_port"}`` has its
MLA KV, indexer K and MTP-layer KV gathered into a per-rank staging buffer
when its prefill finishes, then written to the decode node over RDMA using
TileRT's own wire protocol, layout and transport (``pip install --no-deps
tilert``). Each TP rank ships the layers ``lid % TILERT_PD_SENDERS == rank``.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, ReqToTokenPool

logger = logging.getLogger(__name__)

_TRANSIENT_REJECTS = frozenset({"busy", "cancelling"})
_ADMISSION_ATTEMPTS = 5
_ADMISSION_BACKOFF_S = 0.2
_FLASHMLA_FP8_BPT = 656

_sender: Optional[TileRTKVSender] = None


def kv_transfer_params_of(req: Req) -> Optional[dict]:
    params = (req.sampling_params.custom_params or {}).get("kv_transfer_params")
    if isinstance(params, dict) and params.get("tilert_host"):
        return params
    return None


def initialize(target_pool, draft_pool, max_seq_len) -> None:
    """Called only at scheduler startup, after explicit server opt-in."""
    global _sender
    _sender = TileRTKVSender(target_pool, draft_pool, max_seq_len)


def get_sender() -> Optional[TileRTKVSender]:
    return _sender


def transfer_pending(req: Req) -> bool:
    return _sender is not None and _sender.req is req and _sender.submitted


def can_admit(req: Req) -> bool:
    if kv_transfer_params_of(req) is None:
        return True
    return _sender is not None and (_sender.req is None or _sender.req is req)


def validate_request(req: Req) -> Optional[str]:
    if kv_transfer_params_of(req) is None:
        return None
    if _sender is None:
        return "TileRT KV transfer requires --enable-tilert on the server"
    if req.sampling_params.max_new_tokens != 1:
        return "TileRT prefill requests require max_new_tokens=1"
    if req.session is not None or req.beam_group is not None:
        return "TileRT prefill does not support sessions or beam search"
    params = kv_transfer_params_of(req)
    if not isinstance(params["tilert_host"], str):
        return "tilert_host must be a string"
    try:
        port = int(params.get("tilert_ctrl_port", 5556))
        if not 1 <= port <= 65535:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        return "tilert_ctrl_port must be a port between 1 and 65535"
    return None


def maybe_ship(req: Req, req_to_token_pool: ReqToTokenPool) -> None:
    if kv_transfer_params_of(req) is not None:
        # Intake rejects transfers when disabled. Never load an optional backend
        # or allocate staging memory in response to request-supplied parameters.
        if _sender is None:
            raise RuntimeError("TileRT sender was not initialized")
        _sender.ship(req, req_to_token_pool, kv_transfer_params_of(req))


class TileRTKVSender:
    def __init__(
        self,
        target_pool: DSATokenToKVPool,
        draft_pool: Optional[DSATokenToKVPool],
        max_seq_len: int,
    ):
        import torch
        from tilert.pd_vllm import wire
        from tilert.pd_vllm.profiles import base as profiles
        from tilert.pd_vllm.profiles.mla_nsa import _Reg
        from tilert.pd_vllm.transport import make_transport

        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().attn_tp_size != get_parallel().tp_size:
            raise RuntimeError(
                "TileRT PD needs a full MLA KV replica on every TP rank; "
                "DP attention is not supported"
            )
        self.wire = wire
        self.profile = profiles.get_profile("glm5")
        self.tp_rank = get_parallel().tp_rank
        self.max_seq_len = max_seq_len

        pools = [target_pool] + ([draft_pool] if draft_pool is not None else [])
        mla, ki = [], []
        for pool in pools:
            if not isinstance(pool, DSATokenToKVPool) or pool.page_size != 64:
                raise RuntimeError("TileRT requires a DSA KV pool with 64-token pages")
            for i in range(pool.layer_num):
                lid = len(mla)
                layer_id = pool.start_layer + i
                mla.append((lid, f"L{lid}", pool.kv_buffer[i], 0))
                ki.append(
                    (lid, f"L{lid}", pool.get_index_k_with_scale_buffer(layer_id), 0)
                )
        if len(mla) != self.profile.num_layers:
            raise RuntimeError(
                f"TileRT {self.profile.name} expects {self.profile.num_layers} layers, "
                f"SGLang has {len(mla)}: serve with the MTP draft (EAGLE) or set "
                "TILERT_PD_NO_MTP=1 on both ends"
            )
        strides = {t.shape[-1] * t.element_size() for _, _, t, _ in mla}
        if strides != {_FLASHMLA_FP8_BPT}:
            raise RuntimeError(
                f"MLA cache strides are {strides} B/token, TileRT wants the {_FLASHMLA_FP8_BPT} B "
                "FlashMLA fp8 layout: pass --kv-cache-dtype fp8_e4m3 "
                "--dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv"
            )
        self.reg = _Reg(mla_layers=mla, ki_layers=ki)

        self.req = None
        self.submitted = False
        self.completion = None
        self.completions: queue.Queue = queue.Queue()
        self.senders = len(self.profile.sender_ranks)
        if self.senders > get_parallel().tp_size:
            raise RuntimeError("TILERT_PD_SENDERS cannot exceed the prefill TP size")
        self.is_sender = self.tp_rank in self.profile.sender_ranks
        if not self.is_sender:
            return
        dev = torch.cuda.current_device()
        total = self.profile.staging_bytes(self.reg, self.tp_rank, max_seq_len, 1)
        own = torch.zeros(total, dtype=torch.uint8, device=f"cuda:{dev}")
        self.staging = (
            [own if i == self.tp_rank else None for i in range(self.senders)]
            if self.senders > 1
            else own
        )
        self.transport = make_transport(os.environ.get("TILERT_PD_TRANSPORT"))
        self.transport.init(wire.local_ip())
        self.transport.register(own.data_ptr(), total, dev)
        self.queue: queue.Queue = queue.Queue()
        threading.Thread(
            target=self._loop, name="tilert-kv-sender", daemon=True
        ).start()
        logger.info(
            "TileRT KV sender ready: rank=%d senders=%d staging=%.2f GB transport=%s",
            self.tp_rank,
            self.senders,
            total / 1e9,
            self.transport.name,
        )

    def reserve(self, req: Req) -> bool:
        """Scheduler-owned reservation; never waits for the sender thread."""
        if self.req is not None and self.req is not req:
            return False
        self.req = req
        return True

    def release_unsubmitted(self, req: Req) -> None:
        if self.req is req and not self.submitted:
            self.req = None

    def poll(self):
        """Return (request, error) once *all* TP ranks have finished sending."""
        if not self.submitted:
            if self.req is not None and self.req.finished():
                self.req = None
            return None
        if self.completion is None:
            try:
                self.completion = self.completions.get_nowait()
            except queue.Empty:
                pass
        done = self.completion is not None
        success = not done or self.completion[0] is None
        done, success = self._consensus(done, success)
        if not done:
            return None
        req = self.req
        self.req = None
        self.submitted = False
        self.completion = None
        return req, None if success else "TileRT KV transfer failed"

    @staticmethod
    def _consensus(done, success):
        import torch

        from sglang.srt.runtime_context import get_parallel

        flags = torch.tensor([done, success], dtype=torch.int32)
        if get_parallel().tp_size > 1:
            torch.distributed.all_reduce(
                flags,
                op=torch.distributed.ReduceOp.MIN,
                group=get_parallel().tp_group.cpu_group,
            )
        return flags.tolist()

    def ship(self, req: Req, req_to_token_pool: ReqToTokenPool, params: dict) -> None:
        assert self.req is req and not self.submitted
        self.submitted = True
        if not self.is_sender:
            self.completions.put((None,))
            return
        try:
            seq = len(req.origin_input_ids)
            slots = req_to_token_pool.req_to_token[req.kv.req_pool_idx, :seq:64]
            m = SimpleNamespace(
                rid=self.wire.derive_rid(req.rid),
                num_tokens=seq,
                last_prompt_token=int(req.origin_input_ids[-1]),
                block_ids_per_group=[(slots // 64).tolist()],
                tilert_host=params["tilert_host"],
                tilert_ctrl_port=int(params.get("tilert_ctrl_port", 5556)),
                sampling=params.get("sampling"),
            )
            # Extraction synchronizes before returning, so KV pages can now be
            # released. The reservation protects staging until every rank sends.
            sections = self.profile.extract(
                self.reg, m, self.tp_rank, self.staging, self.max_seq_len
            )
            self.queue.put((m, sections))
        except Exception as exc:
            logger.exception("TileRT KV extraction failed for %s", req.rid)
            self.completions.put((str(exc),))

    def _loop(self) -> None:
        while True:
            m, sections = self.queue.get()
            error = None
            try:
                self._send_with_retry(m, sections)
            except Exception as exc:
                logger.exception("TileRT send failed for %s", m.rid)
                error = str(exc)
            self.completions.put((error,))

    def _send_with_retry(self, m, sections) -> None:
        delay = _ADMISSION_BACKOFF_S
        for attempt in range(_ADMISSION_ATTEMPTS):
            outcome = self._send(m, sections)
            if outcome == "sent":
                return
            if outcome != "transient":
                raise RuntimeError(f"TileRT rejected {m.rid}")
            if attempt + 1 < _ADMISSION_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"TileRT admission retries exhausted for {m.rid}")

    def _send(self, m, sections) -> str:
        wire = self.wire
        seq = sections["seq"]
        t0 = time.time()
        with socket.create_connection((m.tilert_host, m.tilert_ctrl_port), 60) as conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            hello = wire.recv_msg(conn)
            expect = {
                "magic": wire.MAGIC,
                "protocol_version": wire.PROTOCOL_VERSION,
                "layout_version": self.profile.layout_version,
                "senders": self.senders,
                "transport": self.transport.name,
            }
            bad = {k: (hello.get(k), v) for k, v in expect.items() if hello.get(k) != v}
            if bad:
                raise RuntimeError(
                    f"decode handshake mismatch (decode, prefill): {bad}"
                )
            if seq > int(hello["max_seq_len"]):
                raise RuntimeError(
                    f"seq {seq} > decode max_seq_len {hello['max_seq_len']}"
                )
            wire.send_msg(
                conn,
                {
                    "rid": m.rid,
                    "rank": self.tp_rank,
                    "seq_len": seq,
                    "last_prompt_token": m.last_prompt_token,
                    "sampling": m.sampling,
                    "admission_window_s": _ADMISSION_BACKOFF_S
                    * (2 ** (_ADMISSION_ATTEMPTS - 1) - 1),
                },
            )
            ack = wire.recv_msg(conn)
            if not ack.get("accepted"):
                logger.warning(
                    "decode refused %s rank=%d: %s", m.rid, self.tp_rank, ack
                )
                return (
                    "transient"
                    if ack.get("error") in _TRANSIENT_REJECTS
                    else "rejected"
                )
            if ack.get("rid") != m.rid or ack.get("rank") != self.tp_rank:
                raise RuntimeError(f"admission ack does not match {m.rid}: {ack}")
            base = (
                [t.data_ptr() if t is not None else 0 for t in self.staging]
                if isinstance(self.staging, list)
                else self.staging.data_ptr()
            )
            srcs, dsts, lens = self.profile.rdma_plan(
                hello, sections, self.tp_rank, seq, base
            )
            self.transport.write(hello, srcs, dsts, lens)
            wire.send_msg(conn, wire.done_msg(m.rid, self.tp_rank, ack["generation"]))
        logger.info(
            "TileRT sent %s rank=%d seq=%d %.1f MB in %.1f ms",
            m.rid,
            self.tp_rank,
            seq,
            sum(lens) / 1e6,
            1000 * (time.time() - t0),
        )
        return "sent"
