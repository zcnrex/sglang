import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

original = ForwardBatch._compute_mrope_positions_extend
calls = {"fast": 0, "fallback": 0}
enabled = False


def fast(self, runner, batch):
    p = self.positions
    if (
        enabled
        and self.forward_mode.is_extend()
        and self.spec_info is None
        and batch.multimodal_inputs is not None
        and all(mm is None for mm in batch.multimodal_inputs)
        and p is not None
        and p.is_cuda
        and p.dtype == torch.int64
        and p.ndim == 1
        and p.numel() == sum(batch.extend_lens)
    ):
        self.mrope_positions = p.unsqueeze(0).repeat(3, 1)
        calls["fast"] += 1
        return
    calls["fallback"] += 1
    return original(self, runner, batch)


ForwardBatch._compute_mrope_positions_extend = fast
