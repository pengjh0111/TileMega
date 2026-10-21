"""Independent Transformers/torch_npu eager requests using canonical AS-2 tensors."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import types

import torch
import torch_npu
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import StaticCache, DynamicCache

ROOT = Path('/home/d50064638/junhuip')
RUNS = ROOT / 'TileMega/runs/as2'
WEIGHTS = ROOT / 'tilemega-as2/weights'
MODELS = ('qwen3_0_6b', 'llama3_2_1b', 'llama3_2_3b', 'qwen3_1_7b', 'llama3_1_8b')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_model(model_id, attention, norm):
    directory = WEIGHTS / model_id
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['status'] != 'complete' or manifest['model_id'] != model_id:
        raise ValueError('Require a complete canonical tensor manifest')
    cfg = manifest['config']
    config_path = ROOT / 'TileMega/configs/ascend/as2_models' / (model_id + '.json')
    if digest(config_path) != manifest['config_sha256'] or json.loads(config_path.read_text()) != cfg:
        raise ValueError('Config identity mismatch')
    config = AutoConfig.for_model(cfg['model_type'], **{k: v for k, v in cfg.items() if k != 'model_type'})
    config._attn_implementation = attention
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.eval()
    with torch.inference_mode():
        for record in manifest['tensors']:
            file = directory / record['file']
            if file.stat().st_size != record['bytes'] or digest(file) != record['sha256']:
                raise ValueError('Canonical tensor identity mismatch: ' + record['name'])
            # Memory-map one tensor; no full FP32 model or second BF16 device copy.
            cpu = torch.from_file(str(file), shared=False, size=record['bytes'] // 2,
                                  dtype=torch.bfloat16).reshape(record['shape'])
            parent, _, leaf = record['name'].rpartition('.')
            model.get_submodule(parent)._parameters[leaf] = torch.nn.Parameter(cpu.to('npu:0'), requires_grad=False)
            del cpu
        for alias, canonical in manifest['aliases'].items():
            parent, _, leaf = alias.rpartition('.')
            model.get_submodule(parent)._parameters[leaf] = model.get_parameter(canonical)
        # Meta construction leaves deterministic RoPE buffers unmaterialized.
        # Rebuild them with this pinned Transformers implementation.
        for module in model.modules():
            if hasattr(module, 'rope_init_fn'):
                inv_freq, scale = module.rope_init_fn(config, device=torch.device('npu:0'))
                module.register_buffer('inv_freq', inv_freq, persistent=False)
                module.original_inv_freq = inv_freq
                module.attention_scaling = scale
    if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
        raise ValueError('Unmaterialized model state remains')
    if norm == 'fused':
        def fused_norm(module, hidden):
            return torch_npu.npu_rms_norm(hidden, module.weight, module.variance_epsilon)[0]
        count = 0
        for module in model.modules():
            if module.__class__.__name__.endswith('RMSNorm'):
                if not hasattr(module, 'variance_epsilon'):
                    raise ValueError('Unknown RMSNorm epsilon contract')
                module.forward = types.MethodType(fused_norm, module)
                count += 1
        if not count:
            raise ValueError('No RMSNorm modules were adapted')
    token_record = manifest['input_ids']
    token_file = directory / token_record['file']
    if digest(token_file) != token_record['sha256']:
        raise ValueError('Input identity mismatch')
    tokens = torch.from_file(str(token_file), shared=False, size=16 * 64, dtype=torch.int32).reshape(16, 64)
    return model, tokens, manifest, digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 16), required=True)
    parser.add_argument('--attention', choices=('eager', 'sdpa'), default='sdpa')
    parser.add_argument('--cache', choices=('static', 'dynamic'), default='static')
    parser.add_argument('--norm', choices=('hf', 'fused'), default='hf')
    parser.add_argument('--outputs', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.absolute()
    if out.resolve() != out or not out.is_relative_to(RUNS) or out == RUNS:
        parser.error('Use a canonical AS-2 runs output directory')
    if not 2 <= args.outputs <= 1024 or not 0 <= args.warmup <= 1 or not 1 <= args.repetitions <= 3:
        parser.error('Require bounded request/repetition sizes')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') not in {'0', '1'}:
        parser.error('Run through the single-device AS-2 wrapper')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    started = time.monotonic()
    model, tokens_cpu, manifest, weight_manifest_hash = load_model(args.model, args.attention, args.norm)
    batch = args.batch
    tokens = tokens_cpu[:batch].to('npu:0')
    positions = torch.arange(1088, dtype=torch.int64, device='npu:0')
    cache = StaticCache(config=model.config, max_batch_size=batch, max_cache_len=1088,
                        device=torch.device('npu:0'), dtype=torch.bfloat16) if args.cache == 'static' else None
    # Allocate result storage and all position tensors outside request timing.
    outputs = torch.empty((batch, args.outputs), dtype=torch.int32, device='npu:0')
    torch.npu.synchronize()
    identity = dict(model_id=args.model, batch=batch, isl=64, osl=args.outputs,
        capacity=1088, dtype='bf16', seed=manifest['seed'], generator=manifest['generator'],
        config_sha256=manifest['config_sha256'], weight_manifest_sha256=weight_manifest_hash,
        input_sha256=manifest['input_ids']['sha256'], system='torch_npu_eager',
        torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
        transformers_version=__import__('transformers').__version__, attention=args.attention,
        cache=args.cache, norm=args.norm, physical_device_id=int(os.environ['ASCEND_RT_VISIBLE_DEVICES']),
        logical_device_id=0, setup_seconds=time.monotonic()-started,
        status='running', warmup=args.warmup, repetitions=args.repetitions,
        numerical_admission='not_compared', gate='not_started')
    receipt = out / 'result.json'
    receipt.write_text(json.dumps(identity, indent=2)+'\n')
    samples = []
    with torch.inference_mode():
        for repetition in range(-args.warmup, args.repetitions):
            if cache is not None:
                cache.reset()
            state = cache if cache is not None else DynamicCache()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            result = model(input_ids=tokens, past_key_values=state, use_cache=True,
                           cache_position=positions[:64], position_ids=positions[None, :64],
                           logits_to_keep=1, return_dict=True)
            token = result.logits[:, -1].argmax(dim=-1, keepdim=True).to(torch.int32)
            outputs[:, :1].copy_(token)
            torch.npu.synchronize()
            first = time.perf_counter()
            for step in range(1, args.outputs):
                position = 63 + step
                result = model(input_ids=token, past_key_values=state, use_cache=True,
                               cache_position=positions[position:position+1],
                               position_ids=positions[None, position:position+1],
                               logits_to_keep=1, return_dict=True)
                token = result.logits[:, -1].argmax(dim=-1, keepdim=True).to(torch.int32)
                outputs[:, step:step+1].copy_(token)
            torch.npu.synchronize()
            last = time.perf_counter()
            sample = dict(repetition=repetition, warmup=repetition < 0, output_count=args.outputs,
                decode_forwards=args.outputs-1, ttft_ms=(first-t0)*1000,
                e2e_ms=(last-t0)*1000, tpot_ms=(last-first)*1000/(args.outputs-1),
                output_tokens_per_second=batch*args.outputs/(last-t0),
                output_sha256=tensor_hash(outputs), logits_last_sha256=tensor_hash(result.logits),
                nonfinite_logits=int((~torch.isfinite(result.logits)).sum().cpu()))
            if sample['nonfinite_logits']:
                raise ValueError('Nonfinite baseline output')
            samples.append(sample)
            with (out / 'samples.jsonl').open('a') as stream:
                stream.write(json.dumps(sample)+'\n')
            # Output copies occur after t_last and before the next request reset.
            (out / ('tokens_'+str(repetition)+'.bin')).write_bytes(outputs.cpu().numpy().tobytes())
            (out / ('logits_last_'+str(repetition)+'.bin')).write_bytes(
                result.logits.cpu().contiguous().view(torch.uint8).numpy().tobytes())
            print(json.dumps(sample), flush=True)
    identity.update(status='completed_not_numerically_admitted', samples=samples,
                    max_memory_allocated_bytes=torch.npu.max_memory_allocated())
    receipt.write_text(json.dumps(identity, indent=2)+'\n')


if __name__ == '__main__':
    main()
