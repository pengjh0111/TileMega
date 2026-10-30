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
    if len(logits) not in (64,128) or any(not math.isfinite(v) for v in logits):
        raise ValueError('routing audit requires 64/128 finite HF logits')
    for ids in (native, hf):
        if len(ids) != 8 or len(set(ids)) != 8 or any(type(v) is not int or not 0 <= v < len(logits) for v in ids):
            raise ValueError('routing audit requires eight distinct expert IDs within the expert table')
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


def attribute_precursors(mismatches):
    positions = {(r['step'], r['layer'], r['token']) for r in mismatches}
    earlier = {(r['step'], r['token']) for r in mismatches}
    counts = dict(precursor_lower_layer=0, precursor_earlier_step=0, none=0)
    for row in mismatches:
        if row['status'] != 'failure':
            continue
        step, layer, token = (row[k] for k in ('step', 'layer', 'token'))
        lower = any((step, l, token) in positions for l in range(layer))
        previous = any((s, token) in earlier for s in range(step))
        precursor = ('precursor_lower_layer' if lower else
                     'precursor_earlier_step' if previous else 'none')
        row.update(precursor=precursor, precursor_lower_layer=lower, precursor_earlier_step=previous)
        counts[precursor] += 1
    return counts


def bf16_ulp(reference):
    import torch
    _, exponent = torch.frexp(reference.float().abs())
    exponent = torch.where(reference == 0, -133, (exponent - 8).clamp(min=-133))
    return torch.ldexp(torch.ones_like(reference, dtype=torch.float32), exponent)


def distribution(values):
    import torch
    values = values.float().reshape(-1)
    if not len(values):
        return dict(count=0, maximum=None, p50=None, p90=None, p99=None)
    if not bool(torch.isfinite(values).all()):
        raise ValueError('nonfinite routing diagnostic')
    return dict(count=len(values), maximum=values.max().item(),
                **{name: torch.quantile(values, q).item() for name, q in
                   (('p50', .5), ('p90', .9), ('p99', .99))})


def read_native_buffers(engine, layers, batch, hidden, experts):
    import torch
    read = engine.decode_lib.lib.tm_plan_read_buffer_named
    read.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_uint64]
    read.restype = C.c_int
    result = {}
    shapes = dict(route_debug=(64, layers, batch, 8), route_input=(64, layers, batch, hidden),
                  route_logits=(64, layers, batch, experts), route_weights=(64, layers, batch, 8),
                  router_logits=(batch, experts))
    for name, shape in shapes.items():
        count = math.prod(shape)
        raw = ((C.c_uint32 if name == 'route_debug' else C.c_uint16) * count)()
        if read(engine.decode.handle, ('mx.' + name).encode(), raw, C.sizeof(raw)):
            raise RuntimeError('native diagnostic-buffer read failed: ' + name)
        value = torch.tensor(list(raw), dtype=torch.int64)
        result[name] = (value if name == 'route_debug' else value.to(torch.uint16).view(torch.bfloat16)).reshape(shape)
    if not torch.equal(result['route_logits'][-1, -1].view(torch.int16), result['router_logits'].view(torch.int16)):
        raise ValueError('captured final logits differ from mx.router_logits')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'prefill-so', 'decode-so', 'prompt-ids', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    p.add_argument('--mode', choices=('tf', 'free'), default='free')
    p.add_argument('--capture-only', action='store_true')
    p.add_argument('--experts-implementation', choices=('grouped_mm', 'eager'), default='grouped_mm')
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        p.error('run through scripts/mx_gpu_job.sh with NEEDS_MIB=40960')
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(str(a.decode_so) + '.plan.json').read_text())
    if manifest.get('program_kind') != 'moe' or not manifest.get('diag') or manifest.get('executor') != 'mx_static':
        raise ValueError('routing audit requires a diagnostic native MoE decode plan')
    layers = manifest['layers']
    if (layers,manifest['experts']) not in ((24,128),(16,64)) or manifest['top_k'] != 8:
        raise ValueError('routing audit requires the specified MoE-L24 geometry')
    from .plan_pair import validate_plan_pair
    prefill = json.loads(Path(str(a.prefill_so) + '.plan.json').read_text())
    pair_contract = validate_plan_pair(prefill, manifest, a.batch)
    routing = next(r for r in manifest['buffers'] if r['name'] == 'mx.routing_weights')
    if routing['dtype'] != 'bf16':
        raise ValueError('MX-6 native routing capture requires the BF16 contract')
    import torch
    from transformers import AutoModelForCausalLM
    from tilemega.serving.engine import ServingEngine
    from tilemega.serving.workload import validate_prompt_ids
    ids = json.loads(a.prompt_ids.read_text())
    if validate_prompt_ids(ids, a.batch) != 64:
        raise ValueError('routing audit requires 64-token prompts')
    hidden = json.loads((a.model / 'config.json').read_text())['hidden_size']
    with ServingEngine(a.model, a.prefill_so, a.decode_so, a.batch, prompt_len=64,
            max_new_tokens=65, mode='L1', decode_loop=True, prefill_mode='L1', step_events=False) as engine:
        generation = engine.generate(torch.tensor(ids[:a.batch], dtype=torch.int32), 65)
        execution = engine.decode.execution_identity(1, loop=True)
        buffers = read_native_buffers(engine, layers, a.batch, hidden, manifest['experts'])
        native = buffers['route_debug']
        generated = generation.tokens.tolist()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    (a.out / 'native_routes.json').write_text(json.dumps(native.tolist(), separators=(',', ':')) + '\n')
    (a.out / 'generated.json').write_text(json.dumps(generated) + '\n')
    torch.save(buffers, a.out / 'native_diagnostics.pt')
    if a.capture_only:
        (a.out / 'capture.json').write_text(json.dumps(dict(batch=a.batch, layers=layers, steps=64,
            contract='bf16', shapes={k: list(v.shape) for k, v in buffers.items()},
            execution_identity=execution, route_logits_matches_final_scratch=True,
            binary_sha256=hashlib.sha256(a.decode_so.read_bytes()).hexdigest()), indent=2) + '\n')
        return
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', trust_remote_code=False, experts_implementation=a.experts_implementation).cuda().eval()
    if getattr(model.config, '_experts_implementation', None) != a.experts_implementation:
        raise ValueError('HF did not use the requested routing-reference experts')
    if len(model.model.layers) != layers:
        raise ValueError('HF layer count differs from the native plan')
    mismatch, exact, hooks = [], 0, []
    captures = {}
    logit_errors, weight_errors, weight_relative = [], [], []

    def compare(step, layer, batch, logits, selected, weights=None):
        nonlocal exact
        actual_ids = native[step, layer, batch].tolist()
        reference_ids = selected.tolist()
        comparison = compare_routes(actual_ids, reference_ids, logits.tolist())
        if comparison['status'] == 'match':
            exact += 1
            if weights is not None:
                reference = weights[[reference_ids.index(e) for e in actual_ids]].float()
                actual = buffers['route_weights'][step, layer, batch].float()
                delta = (actual - reference).abs()
                weight_errors.append(delta / bf16_ulp(reference))
                weight_relative.append(torch.where(reference != 0, delta / reference.abs(),
                                                   torch.where(delta == 0, 0., float('inf'))))
        else:
            mismatch.append(dict(step=step, layer=layer, token=batch, **comparison))

    def make_hook(layer):
        def hook(_module, _inputs, output):
            if layer in captures or not isinstance(output, tuple) or len(output) != 3:
                raise ValueError('unexpected HF gate hook call or output signature')
            logits, _scores, selected = output
            if logits.dtype != torch.bfloat16 or tuple(logits.shape) != (128, manifest['experts']) or tuple(selected.shape) != (128, 8):
                raise ValueError('HF gate output does not cover the teacher-forced positions')
            captures[layer] = (logits[64:128].float().cpu(), selected[64:128].cpu())
        return hook

    if a.mode == 'tf':
        for layer, block in enumerate(model.model.layers):
            x = buffers['route_input'][:, layer].cuda()
            with torch.inference_mode():
                logits, weights, selected = block.mlp.gate(block.post_attention_layernorm(x))
            if logits.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16 or tuple(logits.shape) != (64 * a.batch, manifest['experts']):
                raise ValueError('HF gate changed the layer teacher-forcing contract')
            logits = logits.cpu().reshape(64, a.batch, manifest['experts'])
            weights = weights.cpu().reshape(64, a.batch, 8)
            selected = selected.cpu().reshape(64, a.batch, 8)
            actual = buffers['route_logits'][:, layer].float()
            logit_errors.append(((actual - logits.float()).abs() / bf16_ulp(logits)).reshape(-1))
            for step in range(64):
                for batch in range(a.batch):
                    compare(step, layer, batch, logits[step, batch], selected[step, batch], weights[step, batch])
            del x
    else:
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
                        compare(step, layer, batch, logits[step], selected[step])
        finally:
            for hook in hooks:
                hook.remove()
    failures = [row for row in mismatch if row['status'] == 'failure']
    precursors = attribute_precursors(mismatch) if a.mode == 'free' else None
    diagnostics = None
    if a.mode == 'tf':
        diagnostics = dict(logit_error_bf16_ulp=distribution(torch.cat(logit_errors)),
            matching_set_weight_error_bf16_ulp=distribution(torch.cat(weight_errors) if weight_errors else torch.empty(0)),
            matching_set_weight_relative_error=distribution(torch.cat(weight_relative) if weight_relative else torch.empty(0)))
    report = dict(model=str(a.model), batch=a.batch, mode=a.mode, contract='bf16', steps=64, layers=layers, positions=64 * layers * a.batch,
        exact_sets=exact, boundary_ties=len(mismatch) - len(failures), failures=len(failures),
        mismatches=mismatch, execution_identity=execution, plan_pair_contract=pair_contract,
        passed=not failures, timing_eligible=False,
        gate_required=a.mode == 'tf', precursor_counts=precursors, diagnostics=diagnostics,
        reference_experts_implementation=a.experts_implementation,
        tie_rule='gap <= 2 * BF16_ULP(abs(HF rank8)); subnormal spacing 2^-133',
        decode_binary_sha256=hashlib.sha256(a.decode_so.read_bytes()).hexdigest())
    (a.out / 'routing_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('mismatches', 'execution_identity')}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
