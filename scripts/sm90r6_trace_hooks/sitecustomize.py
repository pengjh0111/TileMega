"""I-1 worker-side profiler range; enabled only by TILEMEGA_I1_ENGINE.

The baseline and engine files stay unchanged. Python's inherited import path
installs this hook in the actual GPU worker, including spawned processes.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

TARGETS = {
    'vllm': ('vllm.v1.worker.gpu_model_runner', 'GPUModelRunner', 'execute_model'),
    'sglang': ('sglang.srt.managers.tp_worker', 'TpModelWorker', 'forward_batch_generation'),
    'trtllm': ('tensorrt_llm._torch.pyexecutor.model_engine', 'PyTorchModelEngine', 'forward'),
}


def is_decode(engine, instance, args, kwargs):
    if engine == 'vllm':
        batch = args[0] if args else kwargs['scheduler_output']
        counts = batch.num_scheduled_tokens.values()
        return bool(batch.total_num_scheduled_tokens) and all(n == 1 for n in counts)
    if engine == 'sglang':
        batch = args[0] if args else kwargs.get('model_worker_batch')
        if batch is None:
            batch = args[1] if len(args) > 1 else kwargs['forward_batch']
        return batch.forward_mode.is_decode()
    batch = args[0] if args else kwargs['scheduled_requests']
    return not instance.in_warmup and not batch.context_requests and bool(batch.generation_requests)


class Capture:
    def __init__(self, engine, directory, api=None):
        self.engine, self.directory, self.api = engine, Path(directory), api
        self.step = 0
        self.active = self.done = False

    def record(self, event):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory/f'worker-{os.getpid()}.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=event, engine=self.engine, pid=os.getpid(),
                                         decode_step=self.step, host_ns=time.time_ns()))+'\n')

    def before(self, decode):
        if self.done:
            return
        if not decode:
            if self.active:
                raise RuntimeError('I-1 generation ended before the 64-step capture completed')
            self.step = 0
            return
        self.step += 1
        if self.step not in (257, 321):
            return
        if self.api is None:
            import torch
            self.api = torch.cuda
        # Stop before the next forward, retaining sampling and all other GPU
        # work after the 64th forward. Synchronize only at the two outer ends.
        self.api.synchronize()
        if self.step == 257:
            status = self.api.cudart().cudaProfilerStart()
            if status != 0:
                raise RuntimeError(f'cudaProfilerStart failed: {status}')
            self.active = True
            self.record('start')
        else:
            status = self.api.cudart().cudaProfilerStop()
            if status != 0:
                raise RuntimeError(f'cudaProfilerStop failed: {status}')
            self.active, self.done = False, True
            self.record('stop')


def install(module, engine, directory):
    _, class_name, method = TARGETS[engine]
    cls = getattr(module, class_name)
    original = getattr(cls, method)
    capture = Capture(engine, directory)
    capture.record('installed')

    @functools.wraps(original)
    def wrapped(instance, *args, **kwargs):
        capture.before(is_decode(engine, instance, args, kwargs))
        return original(instance, *args, **kwargs)
    setattr(cls, method, wrapped)


class HookFinder(importlib.abc.MetaPathFinder):
    def __init__(self, engine, directory):
        self.engine, self.directory = engine, directory

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGETS[self.engine][0]:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'I-1 cannot find its worker module {fullname}')
        original_loader = spec.loader
        engine, directory = self.engine, self.directory

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return original_loader.create_module(spec)

            def exec_module(self, module):
                original_loader.exec_module(module)
                install(module, engine, directory)
        spec.loader = Loader()
        return spec


engine = os.environ.get('TILEMEGA_I1_ENGINE')
if engine:
    if engine not in TARGETS:
        raise RuntimeError(f'unsupported I-1 engine {engine}')
    sys.meta_path.insert(0, HookFinder(engine, os.environ['TILEMEGA_I1_CAPTURE_DIR']))
