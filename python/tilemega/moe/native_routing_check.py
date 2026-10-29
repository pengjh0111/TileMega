"""Native MoE routing audit and the prompt's two-BF16-ULP boundary rule."""
from __future__ import annotations

import argparse
import ctypes as C
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct


def _ordered_bf16(value):
    if not math.isfinite(value):
        raise ValueError('routing reference logit must be finite')
    bits, = struct.unpack('<I', struct.pack('<f', value))
    if bits & 0xffff:
        raise ValueError('routing reference logit must be an exact BF16 value')
    bits >>= 16
    return (~bits & 0xffff) if bits & 0x8000 else bits | 0x8000


def compare_routes(native, hf, logits):
    if len(logits) != 128 or any(not math.isfinite(v) for v in logits):
        raise ValueError('routing audit requires 128 finite HF logits')
    for ids in (native, hf):
        if len(ids) != 8 or len(set(ids)) != 8 or any(type(v) is not int or not 0 <= v < 128 for v in ids):
            raise ValueError('routing audit requires eight distinct expert IDs in [0,128)')
    if set(native) == set(hf):
        return {'status': 'match'}
    ordered = sorted(logits, reverse=True)
    eighth, ninth = ordered[7:9]
    distance = abs(_ordered_bf16(eighth) - _ordered_bf16(ninth))
    # ULP is the BF16 spacing at the HF eighth logit's magnitude; subnormals
    # retain their fixed spacing. Also report code distance at binade edges.
    ulp = math.ldexp(1.0, max(-133, math.frexp(abs(eighth))[1] - 8)) if eighth else math.ldexp(1.0, -133)
    return dict(status='tie' if eighth - ninth <= 2 * ulp else 'failure', native=native, hf=hf,
        hf_eighth=eighth, hf_ninth=ninth, gap=eighth - ninth, bf16_ulp=ulp,
        gap_in_bf16_ulp=(eighth - ninth) / ulp, bf16_code_distance=distance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'prefill-so', 'decode-so', 'prompt-ids', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch', type=int, choices=(1, 16), required=True)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        p.error('run through scripts/mx_gpu_job.sh with NEEDS_MIB=40960')
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(str(a.decode_so) + '.plan.json').read_text())
    if manifest.get('program_kind') != 'moe' or not manifest.get('diag') or manifest.get('executor') != 'mx_static':
        raise ValueError('routing audit requires a diagnostic native MoE decode plan')
    layers = manifest['layers']
    if layers != 24 or manifest['top_k'] != 8 or manifest['experts'] != 128:
        raise ValueError('routing audit requires the specified MoE-L24 geometry')
    from .plan_pair import validate_plan_pair
    prefill = json.loads(Path(str(a.prefill_so) + '.plan.json').read_text())
    pair_contract = validate_plan_pair(prefill, manifest, a.batch)
    import torch
    from transformers import AutoModelForCausalLM
    from tilemega.serving.engine import ServingEngine
    from tilemega.serving.workload import validate_prompt_ids
    ids = json.loads(a.prompt_ids.read_text())
    if validate_prompt_ids(ids, a.batch) != 64:
        raise ValueError('routing audit requires 64-token prompts')
    with ServingEngine(a.model, a.prefill_so, a.decode_so, a.batch, prompt_len=64,
            max_new_tokens=65, mode='L1', decode_loop=True, prefill_mode='L1', step_events=False) as engine:
        generation = engine.generate(torch.tensor(ids[:a.batch], dtype=torch.int32), 65)
        execution = engine.decode.execution_identity(1, loop=True)
        read = engine.decode_lib.lib.tm_plan_read_buffer
        read.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint64]
        read.restype = C.c_int
        routes = (C.c_uint32 * (64 * layers * a.batch * 8))()
        if read(engine.decode.handle, engine.decode_lib.info.buffer_count, routes, C.sizeof(routes)):
            raise RuntimeError('native routing debug-buffer read failed')
        native = torch.tensor(list(routes), dtype=torch.int64).reshape(64, layers, a.batch, 8)
        generated = generation.tokens.tolist()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    (a.out / 'native_routes.json').write_text(json.dumps(native.tolist(), separators=(',', ':')) + '\n')
    (a.out / 'generated.json').write_text(json.dumps(generated) + '\n')
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', trust_remote_code=False, experts_implementation='grouped_mm').cuda().eval()
    if len(model.model.layers) != layers:
        raise ValueError('HF layer count differs from the native plan')
    mismatch, exact, hooks = [], 0, []
    captures = {}

    def make_hook(layer):
        def hook(_module, _inputs, output):
            if layer in captures or not isinstance(output, tuple) or len(output) != 3:
                raise ValueError('unexpected HF gate hook call or output signature')
            logits, _scores, selected = output
            if logits.dtype != torch.bfloat16 or tuple(logits.shape) != (128, 128) or tuple(selected.shape) != (128, 8):
                raise ValueError('HF gate output does not cover the teacher-forced positions')
            captures[layer] = (logits[64:128].float().cpu(), selected[64:128].cpu())
        return hook

    for layer, block in enumerate(model.model.layers):
        hooks.append(block.mlp.gate.register_forward_hook(make_hook(layer)))
    try:
        for batch in range(a.batch):
            captures.clear()
            whole = torch.tensor([ids[batch] + generated[batch]], dtype=torch.long, device='cuda')
            with torch.inference_mode():
                result = model(whole[:, :128], use_cache=False)
            del result, whole
            if len(captures) != layers:
                raise ValueError('HF did not visit every gate exactly once')
            for layer in range(layers):
                logits, selected = captures[layer]
                for step in range(64):
                    comparison = compare_routes(native[step, layer, batch].tolist(), selected[step].tolist(), logits[step].tolist())
                    if comparison['status'] == 'match':
                        exact += 1
                    else:
                        mismatch.append(dict(step=step, layer=layer, token=batch, **comparison))
    finally:
        for hook in hooks:
            hook.remove()
    failures = [row for row in mismatch if row['status'] == 'failure']
    report = dict(model=str(a.model), batch=a.batch, steps=64, layers=layers, positions=64 * layers * a.batch,
        exact_sets=exact, boundary_ties=len(mismatch) - len(failures), failures=len(failures),
        mismatches=mismatch, execution_identity=execution, plan_pair_contract=pair_contract,
        passed=not failures, timing_eligible=False,
        tie_rule='gap <= 2 * BF16_ULP(abs(HF rank8)); subnormal spacing 2^-133',
        decode_binary_sha256=hashlib.sha256(a.decode_so.read_bytes()).hexdigest())
    (a.out / 'routing_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('mismatches', 'execution_identity')}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
