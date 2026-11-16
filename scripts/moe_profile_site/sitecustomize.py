"""Opt-in diagnostic import hook;never edits an installed framework package."""
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys


class Loader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        cls = module.Qwen3MoeForCausalLM
        original = cls.__init__

        def initialize(instance, *args, **kwargs):
            original(instance, *args, **kwargs)
            from moe_profile_instrumentation import install_nvtx_vllm
            proof = install_nvtx_vllm(instance)
            target = Path(os.environ['MOE_PROFILE_NVTX_OUT'])
            target.mkdir(parents=True, exist_ok=True)
            (target / f'pid{os.getpid()}.json').write_text(json.dumps(proof, indent=2) + '\n')

        cls.__init__ = initialize

    def __getattr__(self, name):
        return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'sglang.srt.models.qwen3_moe':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = Loader(spec.loader)
        return spec


if os.environ.get('MOE_PROFILE_NVTX_OUT'):
    sys.meta_path.insert(0, Finder())
