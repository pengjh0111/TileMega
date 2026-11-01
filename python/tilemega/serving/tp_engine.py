"""Single-process Megatron tensor parallel driver for native dense plans (SM90R5 B7).

One process drives T devices: one weight set, one request state and one plan per
rank, all created from the same generated library (so every rank has the same
grid and the same o/down task order, the deadlock-freedom precondition of plan
§3.5 C3). Each rank's all-reduce receive region and argmax exchange block are
allocated (zeroed) on that rank's device by this harness; the plans only hold
pointers to them.

Timing follows ServingEngine's prompt mode: rank 0's stream records the start
and the TTFT boundary, and TPOT comes from rank 0's per-step device timestamps.
Every launch is preceded by a check that all ranks' plans are at the same
iteration (U17), and every generation checks that all ranks wrote identical
tokens (C4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import time

import torch

from .engine import Generation
from .plan import PlanLibrary
from .state import allocate_state
from .weights import load_weights

L1 = 1


@dataclass
class TpGeneration(Generation):
    # Per rank: device-timestamp span (first, last) of the decode launches, in
    # ns of that device's %globaltimer (U16: the ranks' starting offsets).
    rank_spans: list[tuple[int, int]] = field(default_factory=list)


class _PromptModeOnly:
    """The TP engine has no prefill plan; measurement code may still ask."""
    @staticmethod
    def execution_identity(*_):
        return None


class TpServingEngine:
    def __init__(self, model_dir: str | Path, decode_so: str | Path, batch: int,
                 prompt_len: int = 64, max_new_tokens: int = 1024,
                 devices: tuple[int, ...] | list[int] = (0, 1), decode_loop: bool = True,
                 step_events: bool = True):
        devices = list(devices)
        world = len(devices)
        if prompt_len < 1 or max_new_tokens < 1:
            raise ValueError('prompt length and output length must be positive')
        if len(set(devices)) != world or world < 2:
            raise ValueError('tensor parallel serving needs at least two distinct devices')
        self.library = PlanLibrary(decode_so)
        if self.library.tp_world != world:
            raise ValueError(f'plan is sharded {self.library.tp_world} ways; {world} devices given')
        for a in devices:
            for b in devices:
                if a != b and not torch.cuda.can_device_access_peer(a, b):
                    raise RuntimeError(f'device {a} cannot access device {b} (DT-B1)')
        # One persistent grid per rank; identical grids are a §3.5 C3 premise.
        sms = {torch.cuda.get_device_properties(d).multi_processor_count for d in devices}
        if len(sms) != 1:
            raise RuntimeError('tensor parallel ranks must have identical SM counts')
        info = self.library.info
        if info.seq != 1 or prompt_len + max_new_tokens > info.capacity:
            raise ValueError('plan metadata does not cover this generation')
        self.devices, self.world = devices, world
        self.batch, self.prompt_len, self.max_new_tokens = batch, prompt_len, max_new_tokens
        self.decode_loop, self.step_events = bool(decode_loop), bool(step_events)
        self.decode_mode = L1
        self.prefill = _PromptModeOnly()
        self.memory_report = None
        self.static_vocab_size = int(json.loads((Path(model_dir) / 'config.json').read_text())['vocab_size'])
        self.weights, self.states, self.plans, self.streams = [], [], [], []
        self.ar, self.exchange = [], []
        try:
            for rank, device in enumerate(devices):
                torch.cuda.set_device(device)
                torch.cuda.init()
                where = torch.device('cuda', device)
                weights = load_weights(model_dir, self.library, device=where, tp=(rank, world))
                state = allocate_state(model_dir, batch, None, self.library, device=where, tp=world)
                pointers = {name: tensor.data_ptr() for name, tensor in weights.items()}
                pointers.update(state.pointers)
                plan = self.library.create(batch, pointers, device)
                self.weights.append(weights)
                self.states.append(state)
                self.plans.append(plan)
                # Harness-owned and zeroed: epoch 0 never matches a live step.
                self.ar.append(torch.zeros(self.library.tp_bytes(0), dtype=torch.uint8, device=where))
                self.exchange.append(torch.zeros(self.library.tp_bytes(1), dtype=torch.uint8, device=where))
                self.streams.append(torch.cuda.Stream(device=where))
            ar = [t.data_ptr() for t in self.ar]
            exchange = [t.data_ptr() for t in self.exchange]
            for rank, plan in enumerate(self.plans):
                plan.bind_tp(rank, ar, exchange)
                # Prompt mode (engine.py): step s runs position s.
                plan.set_prompt_len(prompt_len)
                plan.set_steps(list(range(0, prompt_len + max_new_tokens - 1)))
            for device in devices:
                torch.cuda.synchronize(device)
        except BaseException:
            self.close()
            raise
        # The rank-0 plan stands in for the decode plan in measurement code.
        self.decode = self.plans[0]

    def _launch(self, first: int, count: int) -> None:
        iterations = {plan.iteration[L1] for plan in self.plans}
        if len(iterations) != 1:
            raise RuntimeError(f'tensor parallel ranks diverged in iteration: {sorted(iterations)} (U17)')
        # Each launch only enqueues; the ranks' kernels meet in the first
        # all-reduce, so no rank waits on the host between launches.
        if self.decode_loop:
            for plan, stream in zip(self.plans, self.streams):
                plan.launch_steps(first, count, L1, stream.cuda_stream)
            return
        # One step across every rank before the next step, never a whole rank
        # ahead of the others: a rank's step-s kernel spins in its first
        # all-reduce until every peer's step-s kernel runs, so enqueueing one
        # rank's whole range first fills the driver's per-stream launch queue
        # and blocks the host before the peers are enqueued at all - a deadlock
        # (observed: 1023 steps at TP2 never advanced past the first).
        for step in range(first, first + count):
            for plan, stream in zip(self.plans, self.streams):
                plan.launch(step, L1, stream.cuda_stream)

    def generate(self, prompt_ids: torch.Tensor, new_tokens: int | None = None) -> TpGeneration:
        count = new_tokens or self.max_new_tokens
        if count < 1 or count > self.max_new_tokens:
            raise ValueError('token count outside allocated step ring')
        if tuple(prompt_ids.shape) != (self.batch, self.prompt_len):
            raise ValueError('prompt tensor shape disagrees with plan')
        source = prompt_ids.to(device='cpu').contiguous()
        if source.min().item() < 0 or source.max().item() >= self.static_vocab_size:
            raise ValueError('prompt token outside the static decode vocabulary')
        source = source.to(dtype=torch.int32).pin_memory()
        lead = self.streams[0]
        start = torch.cuda.Event(enable_timing=True)
        boundary = torch.cuda.Event(enable_timing=True)
        final = torch.cuda.Event(enable_timing=True)
        outputs = [torch.empty((self.batch, count), dtype=torch.int32, pin_memory=True)
                   for _ in self.plans]
        wall_start = time.perf_counter()
        for state, stream in zip(self.states, self.streams):
            with torch.cuda.stream(stream):
                state.tokens[:, :self.prompt_len].copy_(source, non_blocking=True)
        with torch.cuda.device(self.devices[0]):
            start.record(lead)
        self._launch(0, self.prompt_len)
        with torch.cuda.device(self.devices[0]):
            boundary.record(lead)
        if count > 1:
            self._launch(self.prompt_len, count - 1)
        with torch.cuda.device(self.devices[0]):
            final.record(lead)
        for state, stream, out in zip(self.states, self.streams, outputs):
            with torch.cuda.stream(stream):
                out.copy_(state.tokens[:, self.prompt_len:self.prompt_len + count], non_blocking=True)
        for stream in self.streams:
            stream.synchronize()
        e2e_ms = (time.perf_counter() - wall_start) * 1e3
        for rank in range(1, self.world):
            if not torch.equal(outputs[0], outputs[rank]):
                raise AssertionError(f'rank {rank} generated tokens differing from rank 0 (C4)')
        step_ms = [start.elapsed_time(boundary)]
        spans = []
        step_ns_read = self.step_events and count > 1 and self.decode_loop
        if step_ns_read:
            ns = self.plans[0].read_step_ns(self.prompt_len, count)
            step_ms.extend((ns[i] - ns[i - 1]) / 1e6 for i in range(1, count))
            for plan in self.plans:
                ends = plan.read_step_ns(self.prompt_len, count)
                spans.append((int(ends[0]), int(ends[-1])))
        return TpGeneration(outputs[0], step_ms[0], step_ms, e2e_ms, self.decode_loop and count > 1,
                            step_ns_read, spans)

    def close(self) -> None:
        for plan in getattr(self, 'plans', []):
            plan.close()
        self.plans = []

    def __enter__(self) -> 'TpServingEngine':
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
