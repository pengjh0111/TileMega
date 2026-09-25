"""One-launch-per-step static-batch serving driver."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch

from .plan import PlanLibrary
from .state import allocate_state
from .weights import load_weights


@dataclass
class Generation:
    tokens: torch.Tensor
    ttft_ms: float
    step_ms: list[float]
    e2e_ms: float


class ServingEngine:
    def __init__(self, model_dir: str | Path, prefill_so: str | Path,
                 decode_so: str | Path, batch: int, prompt_len: int = 64,
                 max_new_tokens: int = 1024, mode: str = "auto",
                 device: int = 0):
        if prompt_len != 64 or max_new_tokens < 1:
            raise ValueError("the solved request uses a 64-token prompt")
        torch.cuda.set_device(device)
        torch.cuda.init()  # Both generated libraries use this primary context.
        self.prefill_lib = PlanLibrary(prefill_so)
        self.decode_lib = PlanLibrary(decode_so)
        if (self.prefill_lib.info.seq != prompt_len or
                self.decode_lib.info.seq != 1 or
                self.prefill_lib.info.capacity != self.decode_lib.info.capacity or
                prompt_len + max_new_tokens > self.decode_lib.info.capacity):
            raise ValueError("plan metadata does not cover this generation")
        self.batch = batch
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.weights = load_weights(model_dir, self.prefill_lib, self.decode_lib,
                                    device=torch.device("cuda", device))
        self.state = allocate_state(model_dir, batch, self.prefill_lib,
                                    self.decode_lib,
                                    device=torch.device("cuda", device))
        pointers = {name: tensor.data_ptr() for name, tensor in self.weights.items()}
        pointers.update(self.state.pointers)
        self.prefill = self.prefill_lib.create(batch, pointers, device)
        try:
            self.decode = self.decode_lib.create(batch, pointers, device)
        except BaseException:
            self.prefill.close()
            raise
        self.prefill.set_steps([0])
        self.decode.set_steps(list(range(prompt_len,
                                         prompt_len + max_new_tokens - 1)))
        if mode == "auto":
            self.prefill_mode = (2 if self.prefill_lib.info.modes & 2 else 1)
            self.decode_mode = (2 if self.decode_lib.info.modes & 2 else 1)
        elif mode == "L1":
            self.prefill_mode = self.decode_mode = 1
        elif mode == "L2":
            self.prefill_mode = self.decode_mode = 2
        else:
            raise ValueError(f"unknown serving mode {mode}")
        if not (self.prefill_lib.info.modes & self.prefill_mode and
                self.decode_lib.info.modes & self.decode_mode):
            raise ValueError("requested mode is not in both plan libraries")

    def generate(self, prompt_ids: torch.Tensor,
                 new_tokens: int | None = None) -> Generation:
        count = new_tokens or self.max_new_tokens
        if count < 1 or count > self.max_new_tokens:
            raise ValueError("token count outside allocated step ring")
        if tuple(prompt_ids.shape) != (self.batch, self.prompt_len):
            raise ValueError("prompt tensor shape disagrees with plan")
        source = prompt_ids.to(dtype=torch.int32, device="cpu").contiguous()
        if not source.is_pinned():
            source = source.pin_memory()
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        boundaries = [torch.cuda.Event(enable_timing=True)
                      for _ in range(count)]
        final = torch.cuda.Event(enable_timing=True)
        cpu_tokens = torch.empty((self.batch, count), dtype=torch.int32,
                                 pin_memory=True)
        wall_start = time.perf_counter()
        self.state.tokens[:, :self.prompt_len].copy_(source, non_blocking=True)
        start.record(stream)
        self.prefill.launch(0, self.prefill_mode, stream.cuda_stream)
        boundaries[0].record(stream)
        for step in range(count - 1):
            self.decode.launch(step, self.decode_mode, stream.cuda_stream)
            boundaries[step + 1].record(stream)
        final.record(stream)
        cpu_tokens.copy_(self.state.tokens[:, self.prompt_len:
                                           self.prompt_len + count],
                         non_blocking=True)
        stream.synchronize()
        e2e_ms = (time.perf_counter() - wall_start) * 1e3
        step_ms = [start.elapsed_time(boundaries[0])]
        step_ms.extend(boundaries[i - 1].elapsed_time(boundaries[i])
                       for i in range(1, count))
        return Generation(cpu_tokens, step_ms[0], step_ms, e2e_ms)

    def close(self) -> None:
        self.decode.close()
        self.prefill.close()

    def __enter__(self) -> "ServingEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
