"""One-launch-per-step static-batch serving driver."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch

from .execution import resolve_execution
from .plan import PlanLibrary
from .state import allocate_state
from .weights import load_weights


@dataclass
class Generation:
    tokens: torch.Tensor
    ttft_ms: float
    step_ms: list[float]
    e2e_ms: float
    decode_loop_used: bool = False
    step_ns_read: bool = False


class ServingEngine:
    def __init__(self, model_dir: str | Path, prefill_so: str | Path,
                 decode_so: str | Path, batch: int, prompt_len: int = 64,
                 max_new_tokens: int = 1024, mode: str = "auto",
                 device: int = 0, decode_loop: bool | str = True,
                 decode_chunk: int | None = None, step_events: bool = True,
                 prefill_mode: str | None = None):
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
        decode_manifest = Path(str(decode_so) + ".plan.json")
        decode_pg = (json.loads(decode_manifest.read_text()).get("pg")
                     if decode_manifest.exists() else None)
        mode, decode_loop, prefill_mode = resolve_execution(
            decode_so, mode, decode_loop, prefill_mode, decode_pg == "pages")
        self.decode_loop = decode_loop and decode_pg == "pages"
        self.step_events = bool(step_events)
        self.decode_chunk = decode_chunk
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
            self.prefill_mode = self.decode_mode = 2
        elif mode == "L1":
            self.prefill_mode = self.decode_mode = 1
        elif mode == "L2":
            self.prefill_mode = self.decode_mode = 2
        else:
            raise ValueError(f"unknown serving mode {mode}")
        if prefill_mode is not None:
            if prefill_mode not in ("L1", "L2"):
                raise ValueError(f"unknown prefill mode {prefill_mode}")
            self.prefill_mode = {"L1": 1, "L2": 2}[prefill_mode]
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
        use_loop = self.decode_mode == 2 and self.decode_loop and count > 1 and bool(
            getattr(self.decode_lib.lib, "tm_plan_launch_steps", None))
        boundaries = [torch.cuda.Event(enable_timing=True)
                      for _ in range(1 if use_loop or not self.step_events else count)]
        final = torch.cuda.Event(enable_timing=True)
        cpu_tokens = torch.empty((self.batch, count), dtype=torch.int32,
                                 pin_memory=True)
        wall_start = time.perf_counter()
        self.state.tokens[:, :self.prompt_len].copy_(source, non_blocking=True)
        start.record(stream)
        self.prefill.launch(0, self.prefill_mode, stream.cuda_stream)
        boundaries[0].record(stream)
        if use_loop:
            chunk = self.decode_chunk or count - 1
            if chunk < 1:
                raise ValueError("decode_chunk must be positive")
            for first in range(0, count - 1, chunk):
                self.decode.launch_steps(first, min(chunk, count - 1 - first),
                                         self.decode_mode, stream.cuda_stream)
        else:
            for step in range(count - 1):
                self.decode.launch(step, self.decode_mode, stream.cuda_stream)
                if self.step_events:
                    boundaries[step + 1].record(stream)
        final.record(stream)
        cpu_tokens.copy_(self.state.tokens[:, self.prompt_len:
                                           self.prompt_len + count],
                         non_blocking=True)
        stream.synchronize()
        e2e_ms = (time.perf_counter() - wall_start) * 1e3
        step_ms = [start.elapsed_time(boundaries[0])]
        step_ns_read = use_loop and self.step_events
        if step_ns_read:
            ns = self.decode.read_step_ns(0, count)
            step_ms.extend((ns[i] - ns[i - 1]) / 1e6
                           for i in range(1, count))
        elif self.step_events:
            step_ms.extend(boundaries[i - 1].elapsed_time(boundaries[i])
                           for i in range(1, count))
        return Generation(cpu_tokens, step_ms[0], step_ms, e2e_ms,
                          use_loop, step_ns_read)

    def close(self) -> None:
        self.decode.close()
        self.prefill.close()

    def __enter__(self) -> "ServingEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
