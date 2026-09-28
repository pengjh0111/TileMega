"""Synthetic serving buffers shared by candidate timing and smoke checks."""
from __future__ import annotations
import torch
from .plan import PlanLibrary

def _external_buffers(plan: PlanLibrary, batch: int,
                      vocab: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260925)
    result = {}
    dtypes = {0: torch.bfloat16, 1: torch.float32, 2: torch.int32}
    for buffer in plan.buffers:
        if buffer.role != 1:
            continue
        elements = buffer.elements_constant + batch * buffer.elements_per_batch
        if buffer.dtype == 2:
            tensor = torch.randint(0, vocab, (elements,), dtype=torch.int32,
                                   device="cuda")
        else:
            tensor = torch.randn(elements, dtype=torch.float32,
                                 device="cuda").mul_(0.02).to(dtypes[buffer.dtype])
        result[buffer.name] = tensor
    return result

