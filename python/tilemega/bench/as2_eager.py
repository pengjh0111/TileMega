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


def load_model(model_id, attention, norm, device='npu:0'):
    if device not in ('npu:0','cpu') or (device=='cpu' and norm!='hf'):
        raise ValueError('Use the admitted NPU baseline or CPU mathematical reference')
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
            model.get_submodule(parent)._parameters[leaf] = torch.nn.Parameter(cpu.to(device), requires_grad=False)
            del cpu
        for alias, canonical in manifest['aliases'].items():
            parent, _, leaf = alias.rpartition('.')
            model.get_submodule(parent)._parameters[leaf] = model.get_parameter(canonical)
        # Meta construction leaves deterministic RoPE buffers unmaterialized.
        # Rebuild them with this pinned Transformers implementation.
        for module in model.modules():
            if hasattr(module, 'rope_init_fn'):
                inv_freq, scale = module.rope_init_fn(config, device=torch.device(device))
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


def last_forward_hooks(model):
    control={'enabled':False};saved={}
    def hook(name,before=False):
        def observe(module,args,result=None):
            if control['enabled']:saved[name]=(args[0] if before else result).detach().clone()
        return observe
    for i,layer in enumerate(model.model.layers):
        for name,module in (('norm1',layer.input_layernorm),('norm2',layer.post_attention_layernorm)):
            module.register_forward_hook(hook(f'l{i}.{name}'))
            module.register_forward_pre_hook(hook(f'l{i}.{name}.input',True))
        layer.self_attn.o_proj.register_forward_pre_hook(hook(f'l{i}.context',True))
        layer.mlp.down_proj.register_forward_pre_hook(hook(f'l{i}.act',True))
        for name,module in [('q.raw',layer.self_attn.q_proj),('k.raw',layer.self_attn.k_proj),('v.raw',layer.self_attn.v_proj),
                ('o.raw',layer.self_attn.o_proj),('gate.raw',layer.mlp.gate_proj),('up.raw',layer.mlp.up_proj),('down.raw',layer.mlp.down_proj)]:
            module.register_forward_hook(hook(f'l{i}.{name}'))
        for name in ('q_norm','k_norm'):
            if hasattr(layer.self_attn,name):getattr(layer.self_attn,name).register_forward_hook(hook(f'l{i}.{name}'))
    model.model.norm.register_forward_hook(hook('final.normalized'))
    return control,saved


def save_last_forward(out,saved,cfg,state,batch,past):
    d=cfg['head_dim'];g=cfg['num_key_value_heads'];ratio=cfg['num_attention_heads']//g;f=cfg['intermediate_size']
    for i in range(cfg['num_hidden_layers']):
        q,k,v=(saved[f'l{i}.{name}.raw'] for name in ('q','k','v'))
        saved[f'l{i}.qkv']=torch.cat((q.reshape(-1,g,ratio,d),k.reshape(-1,g,1,d),v.reshape(-1,g,1,d)),2).reshape(-1,g*(ratio+2)*d)
        gate,up=(saved[f'l{i}.{name}.raw'] for name in ('gate','up'))
        saved[f'l{i}.gate_up.raw']=torch.stack((gate.reshape(-1,f//16,16),up.reshape(-1,f//16,16)),2).reshape(-1,2*f)
    folder=out/'last_forward';folder.mkdir();files=[]
    for name,tensor in saved.items():
        value=tensor.cpu().contiguous();path=folder/(name+'.bin');path.write_bytes(value.view(torch.uint8).numpy().tobytes())
        files.append(dict(name=name,file=path.name,shape=list(value.shape),dtype=str(value.dtype),sha256=digest(path)))
    for layer,(key,value) in enumerate(zip(state.key_cache,state.value_cache)):
        for side,tensor in (('k',key),('v',value)):
            data=tensor[:batch,:,:past+1,:].cpu().contiguous();path=folder/f'l{layer}.kv_{side}.bin'
            path.write_bytes(data.view(torch.uint8).numpy().tobytes())
            files.append(dict(name=f'l{layer}.kv_{side}',file=path.name,shape=list(data.shape),dtype=str(data.dtype),sha256=digest(path)))
    (folder/'capture.json').write_text(json.dumps(dict(scope='independent eager last-forward diagnostic; not performance',
        past=past,seq=1,batch=batch,files=files),indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--device', choices=('npu:0','cpu'), default='npu:0', help='CPU is a correctness reference only')
    parser.add_argument('--batch', type=int, choices=(1, 16), required=True)
    parser.add_argument('--attention', choices=('eager', 'sdpa'), default='sdpa')
    parser.add_argument('--cache', choices=('static', 'dynamic'), default='static')
    parser.add_argument('--norm', choices=('hf', 'fused'), default='hf')
    parser.add_argument('--outputs', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--save-kv-rows', action='store_true', help='Copy final valid KV rows after t_last, for correctness only')
    parser.add_argument('--capture-last-forward',action='store_true',help='One diagnostic request; B1 may capture long-context KV, no performance admission')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.absolute()
    if out.resolve() != out or not out.is_relative_to(RUNS) or out == RUNS:
        parser.error('Use a canonical AS-2 runs output directory')
    if not 2 <= args.outputs <= 1024 or not 0 <= args.warmup <= 1 or not 1 <= args.repetitions <= 3:
        parser.error('Require bounded request/repetition sizes')
    if args.save_kv_rows and args.cache!='static':
        parser.error('KV diagnostics require the fixed-capacity reference cache')
    if args.capture_last_forward and ((args.outputs>128 and args.batch!=1) or args.warmup or args.repetitions!=1 or args.cache!='static'):
        parser.error('Last-forward diagnostic requires warmup0, repetition1, static cache and B1 for outputs>128')
    if args.device=='cpu' and args.norm!='hf':
        parser.error('CPU correctness reference uses independent HF normalization')
    if args.device=='npu:0' and os.environ.get('ASCEND_RT_VISIBLE_DEVICES') not in {'0', '1'}:
        parser.error('Run through the single-device AS-2 wrapper')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if args.device=='npu:0':torch.npu.set_device(0)
    def synchronize():
        if args.device=='npu:0':torch.npu.synchronize()
    started = time.monotonic()
    model, tokens_cpu, manifest, weight_manifest_hash = load_model(args.model, args.attention, args.norm,device=args.device)
    capture,saved=last_forward_hooks(model) if args.capture_last_forward else (None,None)
    rotary={}
    if args.save_kv_rows:
        def capture_rotary(module,inputs,result):rotary['last']=result
        model.model.rotary_emb.register_forward_hook(capture_rotary)
    batch = args.batch
    tokens = tokens_cpu[:batch].to(args.device)
    positions = torch.arange(1088, dtype=torch.int64, device=args.device)
    cache = StaticCache(config=model.config, max_batch_size=batch, max_cache_len=1088,
                        device=torch.device(args.device), dtype=torch.bfloat16) if args.cache == 'static' else None
    # Allocate result storage and all position tensors outside request timing.
    outputs = torch.empty((batch, args.outputs), dtype=torch.int32, device=args.device)
    synchronize()
    identity = dict(model_id=args.model, batch=batch, isl=64, osl=args.outputs,
        capacity=1088, dtype='bf16', seed=manifest['seed'], generator=manifest['generator'],
        config_sha256=manifest['config_sha256'], weight_manifest_sha256=weight_manifest_hash,
        input_sha256=manifest['input_ids']['sha256'], system='torch_npu_eager' if args.device=='npu:0' else 'cpu_bf16_reference',
        torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
        transformers_version=__import__('transformers').__version__, attention=args.attention,
        cache=args.cache, norm=args.norm, physical_device_id=int(os.environ['ASCEND_RT_VISIBLE_DEVICES']) if args.device=='npu:0' else None,
        logical_device_id=0 if args.device=='npu:0' else None, setup_seconds=time.monotonic()-started,
        status='running', warmup=args.warmup, repetitions=args.repetitions,
        diagnostic_capture_last_forward=args.capture_last_forward,
        numerical_admission='not_compared', gate='not_started')
    receipt = out / 'result.json'
    receipt.write_text(json.dumps(identity, indent=2)+'\n')
    samples = []
    with torch.inference_mode():
        for repetition in range(-args.warmup, args.repetitions):
            if cache is not None:
                cache.reset()
            state = cache if cache is not None else DynamicCache()
            synchronize()
            t0 = time.perf_counter()
            result = model(input_ids=tokens, past_key_values=state, use_cache=True,
                           cache_position=positions[:64], position_ids=positions[None, :64],
                           logits_to_keep=1, return_dict=True)
            token = result.logits[:, -1].argmax(dim=-1, keepdim=True).to(torch.int32)
            outputs[:, :1].copy_(token)
            synchronize()
            first = time.perf_counter()
            for step in range(1, args.outputs):
                if capture is not None:capture['enabled']=step==args.outputs-1
                position = 63 + step
                result = model(input_ids=token, past_key_values=state, use_cache=True,
                               cache_position=positions[position:position+1],
                               position_ids=positions[None, position:position+1],
                               logits_to_keep=1, return_dict=True)
                token = result.logits[:, -1].argmax(dim=-1, keepdim=True).to(torch.int32)
                outputs[:, step:step+1].copy_(token)
            synchronize()
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
            if args.save_kv_rows:
                position=62+args.outputs
                for layer,(key,value) in enumerate(zip(state.key_cache,state.value_cache)):
                    for side,tensor in [('k',key),('v',value)]:
                        row=tensor[:batch,:,position,:].cpu().contiguous()
                        (out/f'kv_{side}_l{layer}_r{repetition}.bin').write_bytes(row.view(torch.uint8).numpy().tobytes())
                for side,tensor in zip(('cos','sin'),rotary['last']):
                    (out/f'rope_{side}_r{repetition}.bin').write_bytes(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
            if capture is not None:save_last_forward(out,saved,manifest['config'],state,batch,62+args.outputs)
            print(json.dumps(sample), flush=True)
    identity.update(status='completed_not_numerically_admitted', samples=samples,
                    max_memory_allocated_bytes=torch.npu.max_memory_allocated() if args.device=='npu:0' else None)
    receipt.write_text(json.dumps(identity, indent=2)+'\n')


if __name__ == '__main__':
    main()
