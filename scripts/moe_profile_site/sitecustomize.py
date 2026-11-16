"""Opt-in diagnostic import hook;never edits an installed framework package."""
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import sys

ROUTE_META = dict(armed=False, records=[])


def install_route_metadata(instance):
    import torch
    from sglang.srt.managers.scheduler import Scheduler
    original = instance.forward
    signature = inspect.signature(original)
    from moe_profile_instrumentation import install_sglang, arm as arm_helpers, export as export_helpers
    install_sglang(instance, lambda: ROUTE_META.get('current', dict(is_decode=False, index=-1)))

    def forward(*args, **kwargs):
        if ROUTE_META['armed']:
            values = signature.bind(*args, **kwargs).arguments
            batch = values['forward_batch']
            ROUTE_META['current'] = dict(is_decode=batch.forward_mode.is_decode(),
                                          index=len(ROUTE_META['records']))
            ROUTE_META['records'].append(dict(
                mode=str(batch.forward_mode), is_decode=batch.forward_mode.is_decode(),
                seq_lens=batch.seq_lens.detach().clone(),
                positions=values['positions'].detach().clone(),
                input_shape=list(values['input_ids'].shape)))
        return original(*args, **kwargs)

    def arm(scheduler, parameters):
        ROUTE_META.update(armed=True, records=[])
        arm_helpers()

    def export(scheduler, parameters):
        ROUTE_META['armed'] = False
        target = Path(parameters['path'])
        target.mkdir(parents=True, exist_ok=True)
        records = [{key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in row.items()} for row in ROUTE_META['records']]
        torch.save(dict(diagnostic_only=True, records=records,
                        scope='Actual forward mode, positions and KV sequence lengths;'
                              'device clones during ROI,host copies after ROI.'),
                   target / f'forward_metadata_pid{os.getpid()}.pt')
        export_helpers(target / 'helper_metrics')

    instance.forward = forward
    Scheduler.moe_profile_arm_route_metadata = arm
    Scheduler.moe_profile_export_route_metadata = export


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
            if os.environ.get('MOE_PROFILE_NVTX_OUT'):
                from moe_profile_instrumentation import install_nvtx_vllm
                proof = install_nvtx_vllm(instance)
                from sglang.srt.layers.moe.utils import get_moe_runner_backend
                from moe_profile_benchmark import describe_model
                from moe_profile_instrumentation import set_nvtx_active
                from sglang.srt.managers.scheduler import Scheduler
                start, stop = Scheduler.start_profile, Scheduler.stop_profile

                def start_profile(scheduler, *args, **kwargs):
                    set_nvtx_active(True)
                    return start(scheduler, *args, **kwargs)

                def stop_profile(scheduler, *args, **kwargs):
                    try:
                        return stop(scheduler, *args, **kwargs)
                    finally:
                        set_nvtx_active(False)

                Scheduler.start_profile, Scheduler.stop_profile = start_profile, stop_profile
                set_nvtx_active(False)
                proof['actual_moe_runner_backend'] = str(get_moe_runner_backend())
                proof['runtime_model'] = describe_model(instance)
                target = Path(os.environ['MOE_PROFILE_NVTX_OUT'])
                target.mkdir(parents=True, exist_ok=True)
                (target / f'pid{os.getpid()}.json').write_text(json.dumps(proof, indent=2) + '\n')
            if os.environ.get('MOE_PROFILE_ROUTE_META_OUT'):
                install_route_metadata(instance)

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


if os.environ.get('MOE_PROFILE_NVTX_OUT') or os.environ.get('MOE_PROFILE_ROUTE_META_OUT'):
    sys.meta_path.insert(0, Finder())
