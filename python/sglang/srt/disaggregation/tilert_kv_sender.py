"""Ship a finished GLM-5 prefill's KV to a TileRT decode node.

SGLang-side counterpart of ``tilert.pd_vllm.prefill_connector``: a request
carrying ``kv_transfer_params={"tilert_host", "tilert_ctrl_port"}`` has its
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

import torch

from sglang.srt.runtime_context import get_parallel

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


def maybe_ship(
    req: Req,
    req_to_token_pool: ReqToTokenPool,
    target_pool: DSATokenToKVPool,
    draft_pool: Optional[DSATokenToKVPool],
) -> None:
    params = kv_transfer_params_of(req)
    if params is None:
        return
    global _sender
    if _sender is None:
        _sender = TileRTKVSender(
            target_pool, draft_pool, req_to_token_pool.req_to_token.shape[1]
        )
    try:
        _sender.ship(req, req_to_token_pool, params)
    except Exception:
        logger.exception("TileRT KV extraction failed for %s", req.rid)


class TileRTKVSender:
    def __init__(
        self,
        target_pool: DSATokenToKVPool,
        draft_pool: Optional[DSATokenToKVPool],
        max_seq_len: int,
    ):
        from tilert.pd_vllm import wire
        from tilert.pd_vllm.profiles import base as profiles
        from tilert.pd_vllm.profiles.mla_nsa import _Reg
        from tilert.pd_vllm.transport import make_transport

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
        bpt = mla[0][2].shape[-1] * mla[0][2].element_size()
        if bpt != _FLASHMLA_FP8_BPT:
            raise RuntimeError(
                f"MLA cache is {bpt} B/token, TileRT wants the {_FLASHMLA_FP8_BPT} B "
                "FlashMLA fp8 layout: pass --kv-cache-dtype fp8_e4m3 "
                "--dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv"
            )
        self.reg = _Reg(mla_layers=mla, ki_layers=ki)

        self.senders = len(self.profile.sender_ranks)
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
        # Held from the gather into staging until that request's RDMA write ends.
        self.inflight = threading.Lock()
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

    def ship(self, req: Req, req_to_token_pool: ReqToTokenPool, params: dict) -> None:
        if not self.is_sender:
            return
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
        self.inflight.acquire()
        try:
            # Synchronous gather: the pages may be reused as soon as we return.
            sections = self.profile.extract(
                self.reg, m, self.tp_rank, self.staging, self.max_seq_len
            )
        except BaseException:
            self.inflight.release()
            raise
        self.queue.put((m, sections))

    def _loop(self) -> None:
        while True:
            m, sections = self.queue.get()
            try:
                self._send_with_retry(m, sections)
            finally:
                self.inflight.release()

    def _send_with_retry(self, m, sections) -> None:
        delay = _ADMISSION_BACKOFF_S
        for attempt in range(_ADMISSION_ATTEMPTS):
            try:
                outcome = self._send(m, sections)
            except Exception:
                logger.exception("TileRT send failed for %s", m.rid)
                break
            if outcome != "transient":
                break
            time.sleep(delay)
            delay *= 2
        else:
            logger.error(
                "gave up admitting %s rank=%d; decode will time out waiting",
                m.rid,
                self.tp_rank,
            )

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
