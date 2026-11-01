"""Single-process dense TP: one library, private streams, peer-bound plans."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import torch

from .engine import Generation
from .plan import PlanLibrary
from .state import allocate_tp_state
from .weights import load_weights


@dataclass
class TpGeneration(Generation):
    rank_kernel_ms: list[dict] = field(default_factory=list)


class TpServingEngine:
    def __init__(self, model_dir, decode_so, batch, prompt_len=64,
                 max_new_tokens=1024, devices=(0, 1)):
        self.devices = tuple(map(int, devices))
        self.lib = PlanLibrary(decode_so)
        self.tp = self.lib.tp_world()
        if self.tp not in (2, 4) or len(self.devices) != self.tp or set(self.devices) != set(range(self.tp)):
            raise ValueError('logical devices must cover exactly the TP2/TP4 plan world')
        if (self.lib.info.phase != 1 or self.lib.info.past_lo != 0 or
                prompt_len < 1 or max_new_tokens < 1 or
                prompt_len + max_new_tokens > self.lib.info.capacity or
                prompt_len + max_new_tokens - 2 > self.lib.info.past_hi or
                not self.lib.info.batch_lo <= batch <= self.lib.info.batch_hi):
            raise ValueError('TP decode metadata does not cover prompt-mode generation')
        for source in self.devices:
            for target in self.devices:
                if source != target and not torch.cuda.can_device_access_peer(source, target):
                    raise ValueError(f'P2P unavailable for logical devices {source}->{target}')
        self.batch, self.prompt_len, self.max_new_tokens = batch, prompt_len, max_new_tokens
        self.ranks = []
        self.poisoned = False
        self.last_failure = None
        self.memory_report = None
        try:
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    weights = load_weights(model_dir, self.lib,
                                           device=torch.device('cuda', device), tp=(rank, self.tp))
                    state = allocate_tp_state(model_dir, batch, self.lib,
                                              torch.device('cuda', device), self.tp)
                    pointers = {name: value.data_ptr() for name, value in weights.items()}
                    plan = self.lib.create(batch, {**pointers, **state.pointers}, device)
                    stream = torch.cuda.Stream(device)
                    self.ranks.append(dict(rank=rank, device=device, weights=weights,
                                           state=state, plan=plan, stream=stream))
            buffers = [rank['plan'].tp_buffers() for rank in self.ranks]
            ar = [value[0] for value in buffers]
            exchange = [value[1] for value in buffers]
            for rank in self.ranks:
                with torch.cuda.device(rank['device']):
                    plan = rank['plan']
                    plan.bind_tp(rank['rank'], ar, exchange)
                    plan.set_steps(list(range(prompt_len + max_new_tokens - 1)))
                    plan.set_prompt_len(prompt_len)
                    # Packing and runtime clears precede the private-stream launches.
                    torch.cuda.synchronize(rank['device'])
        except BaseException:
            self.close()
            raise

    def _cohort_iteration(self):
        values = [rank['plan'].iteration[1] for rank in self.ranks]
        if len(set(values)) != 1:
            raise AssertionError(f'TP rank iteration counters differ: {values}')
        return values[0]

    def _failure(self, stage, rank_index, error):
        self.poisoned = True
        watchdogs = []
        for rank in self.ranks:
            with torch.cuda.device(rank['device']):
                try:
                    if rank['stream'].query():
                        rank['stream'].synchronize()
                    watchdog = rank['plan'].watchdog()
                except BaseException as query_error:
                    watchdog = dict(error=repr(query_error))
                watchdogs.append(dict(rank=rank['rank'], watchdog=watchdog,
                                      iteration=rank['plan'].iteration[1]))
        self.last_failure = dict(stage=stage, rank=rank_index, error=repr(error), ranks=watchdogs)

    def generate(self, prompt_ids, new_tokens=None):
        count = self.max_new_tokens if new_tokens is None else int(new_tokens)
        if self.poisoned:
            raise RuntimeError('TP engine is poisoned; restart the process')
        if not 1 <= count <= self.max_new_tokens:
            raise ValueError('requested token count outside configured generation')
        source = torch.as_tensor(prompt_ids, dtype=torch.int32, device='cpu').contiguous()
        if tuple(source.shape) != (self.batch, self.prompt_len):
            raise ValueError('TP prompt shape differs from the configured batch and ISL')
        events = [dict(prompt_start=torch.cuda.Event(enable_timing=True),
                       prompt_end=torch.cuda.Event(enable_timing=True),
                       decode_start=torch.cuda.Event(enable_timing=True),
                       decode_end=torch.cuda.Event(enable_timing=True)) for _ in self.ranks]
        tokens = torch.empty((self.batch, count), dtype=torch.int32, pin_memory=True)
        wall_start = time.perf_counter()
        for rank in self.ranks:
            with torch.cuda.device(rank['device']), torch.cuda.stream(rank['stream']):
                rank['state'].tokens[:, :self.prompt_len].copy_(source, non_blocking=True)
        events[0]['prompt_start'].record(self.ranks[0]['stream'])
        stage, current_rank = 'prompt', -1
        try:
            base = self._cohort_iteration()
            for rank, event in zip(self.ranks, events):
                current_rank = rank['rank']
                # A cohort is checked together; each counter advances on enqueue.
                assert rank['plan'].iteration[1] == base
                with torch.cuda.device(rank['device']):
                    if current_rank:
                        event['prompt_start'].record(rank['stream'])
                    rank['plan'].launch_steps(0, self.prompt_len, 1, rank['stream'].cuda_stream)
                    if current_rank:
                        event['prompt_end'].record(rank['stream'])
            events[0]['prompt_end'].record(self.ranks[0]['stream'])
            self._cohort_iteration()
            if count > 1:
                stage = 'decode'
                base = self._cohort_iteration()
                for rank, event in zip(self.ranks, events):
                    current_rank = rank['rank']
                    assert rank['plan'].iteration[1] == base
                    with torch.cuda.device(rank['device']):
                        event['decode_start'].record(rank['stream'])
                        rank['plan'].launch_steps(self.prompt_len, count - 1, 1,
                                                  rank['stream'].cuda_stream)
                        event['decode_end'].record(rank['stream'])
                self._cohort_iteration()
            root = self.ranks[0]
            with torch.cuda.device(root['device']), torch.cuda.stream(root['stream']):
                tokens.copy_(root['state'].tokens[:, self.prompt_len:self.prompt_len + count],
                             non_blocking=True)
            for rank in self.ranks:
                rank['stream'].synchronize()
        except BaseException as error:
            self._failure(stage, current_rank, error)
            raise
        e2e_ms = (time.perf_counter() - wall_start) * 1e3
        ttft_ms = events[0]['prompt_start'].elapsed_time(events[0]['prompt_end'])
        kernel_ms = []
        for rank, event in zip(self.ranks, events):
            with torch.cuda.device(rank['device']):
                kernel_ms.append(dict(rank=rank['rank'], logical_device=rank['device'],
                    prompt_ms=event['prompt_start'].elapsed_time(event['prompt_end']),
                    decode_ms=event['decode_start'].elapsed_time(event['decode_end']) if count > 1 else 0.))
        for rank in self.ranks:
            other = rank['state'].tokens[:, self.prompt_len:self.prompt_len + count].cpu()
            if not torch.equal(tokens, other):
                error = AssertionError(f'rank {rank["rank"]} generated different tokens')
                self._failure('rank_consistency', rank['rank'], error)
                raise error
        return TpGeneration(tokens=tokens, ttft_ms=ttft_ms, step_ms=[], e2e_ms=e2e_ms,
                            decode_loop_used=True, step_ns_read=False, rank_kernel_ms=kernel_ms)

    def close(self):
        for rank in getattr(self, 'ranks', []):
            with torch.cuda.device(rank['device']):
                if self.poisoned and not rank['stream'].query():
                    # Failed cohorts are process-fatal; do not free live peer buffers.
                    rank['plan'].handle = None
                else:
                    rank['plan'].close()
        self.ranks = []

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
