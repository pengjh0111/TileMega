"""HF auto tensor parallelism with the existing static-cache compile path."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prompt-ids', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--mode', default='reduce-overhead')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from baseline_common import measure
    from measurement_policy import TimingPolicy
    from workload import validate_prompt_ids
    import torch
    import torch.distributed as dist
    import transformers
    from transformers import AutoModelForCausalLM, CompileConfig

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world not in (2, 4):
        raise ValueError('MX7 baseline requires TP2 or TP4')
    torch.cuda.set_device(local_rank)
    # CP0 verified is_nccl_available(); MetaX provides this backend through MCCL.
    dist.init_process_group(backend='nccl')
    ids = json.loads(args.prompt_ids.read_text())
    prompt_len = validate_prompt_ids(ids, args.batch)
    sdp_flags_constant = hasattr(torch, 'disable_flash_attn')
    if sdp_flags_constant:
        for name in ('flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled'):
            setattr(torch.backends.cuda, name,
                    torch._dynamo.assume_constant_result(getattr(torch.backends.cuda, name)))
        torch.disable_flash_attn = True
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, tp_plan='auto').eval()
    model.generation_config.cache_implementation = 'static'
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.generation_config.compile_config = CompileConfig(fullgraph=True, mode=args.mode)
    # HF TP q/k norms install gradient hooks; Dynamo requires this even in inference.
    torch._dynamo.config.compiled_autograd = True
    meta = dict(system='torch.compile', model=str(args.model), tp=world,
                torch_version=torch.__version__, transformers_version=transformers.__version__,
                accelerate_version=importlib.metadata.version('accelerate'),
                compiled_autograd=True,
                fullgraph=True, mode=args.mode, tp_plan='auto', backend='nccl',
                sdp_flags_constant=sdp_flags_constant,
                metax_flash_attn_disabled=sdp_flags_constant,
                compile_path='generation_config.compile_config (decode forward; prefill eager)')

    def run(batch, count):
        input_ids = torch.tensor(ids[:batch], dtype=torch.long,
                                 device=torch.device('cuda', local_rank)).contiguous()
        with torch.inference_mode():
            output = model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                    do_sample=False, min_new_tokens=count, max_new_tokens=count)
        torch.cuda.synchronize()
        if not torch.equal(output[:, :prompt_len].cpu(), input_ids.cpu()):
            raise AssertionError('HF TP prompt order differs')
        meta['compiled_call_used'] = hasattr(model, '_compiled_call')
        if count > 1 and not meta['compiled_call_used']:
            raise RuntimeError('HF TP did not use the requested compiled decode')
        return output[:, prompt_len:].tolist()

    def generate(batch, count):
        command = [(batch, count)]
        dist.broadcast_object_list(command, src=0)
        return run(batch, count)

    try:
        if rank == 0:
            measure(args.out, (args.batch,), args.max_tokens, args.warmup, args.repeats,
                    TimingPolicy(args.policy), generate, meta)
            dist.broadcast_object_list([None], src=0)
        else:
            while True:
                command = [None]
                dist.broadcast_object_list(command, src=0)
                if command[0] is None:
                    break
                run(*command[0])
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
