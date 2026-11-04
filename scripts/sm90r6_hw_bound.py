#!/usr/bin/env python3
"""H3: logical BF16 decode lower bounds from frozen timings and measured H1/H2."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from estimate_r16_mbu import dense_payload
from sm90r4_inputs import model_dir

ROOT = Path(__file__).resolve().parents[1]
# Later rows supersede earlier measurements of the same system/model/TP/B.
SOURCES = [
    ('SM90R3_llm_final.csv', 'TileMega'), ('SM90R4_llm_final.csv', 'TileMega'),
    ('SM90R5_moe_final.csv', 'TileMega'), ('SM90R5_tp_final.csv', 'TileMega'),
    ('R16_final_baselines.csv', ''), ('SM90R4_baselines_new.csv', ''),
    ('SM90R4_mpk.csv', 'mpk'), ('SM90R5_mpk.csv', 'mpk'),
    ('SM90R5_olmoe_torchcompile.csv', ''), ('SM90R5_tp_baselines.csv', ''),
    ('SM90R5_mpk_tp.csv', 'mpk'), ('SM90R6_torchcompile_tp.csv', 'torch.compile')]


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def peaks(path):
    rows = read_csv(path)
    cards = {0, 1, 2, 3, 4, 5, 7}
    seen = set()
    for row in rows:
        key = (int(row['gpu']), row['item'], row['config'])
        if key in seen or key[0] not in cards:
            raise ValueError('duplicate H1 configuration or non-timing card')
        seen.add(key)
        if int(row['processes']) != 5 or int(row['failed_processes']):
            raise ValueError('H1 requires five accepted processes per supported configuration')
        # Idle/application-clock bits are allowed; throttle/slowdown bits are not.
        if any(int(mask, 0) & ~3 for mask in row['throttle_reasons'].split(';')):
            raise ValueError('H1 contains a throttled measurement')
        positive(row['median'], 'H1 peak')
    for gpu in cards:
        for item, count in [('hbm_read', 2), ('hbm_read_bulk', 4),
                            ('bf16_wgmma', 1), ('bf16_mma_sync', 1)]:
            if sum(g == gpu and i == item for g, i, _ in seen) != count:
                raise ValueError(f'incomplete H1 coverage: GPU {gpu}, {item}')
    bandwidth = max(float(r['median']) for r in rows if r['item'].startswith('hbm_read'))
    compute = max(float(r['median']) for r in rows if r['item'] == 'bf16_wgmma')
    return bandwidth, compute


def routing(path):
    grouped = {}
    for row in read_csv(path):
        key = (row['model'], int(row['batch']))
        if row['evidence'] != 'verified' or int(row['steps']) != 1024:
            raise ValueError('H2 needs verified complete trajectories')
        grouped.setdefault(key, []).append(row)
    for (model, batch), rows in grouped.items():
        config = json.loads((model_dir(model) / 'config.json').read_text())
        if sorted(int(r['layer']) for r in rows) != list(range(config['num_hidden_layers'])):
            raise ValueError(f'incomplete or duplicate H2 layers: {model} B{batch}')
        for row in rows:
            count = positive(row['distinct_mean'], 'H2 distinct experts')
            if not config['num_experts_per_tok'] <= count <= min(config['num_experts'],
                                                               batch*config['num_experts_per_tok']):
                raise ValueError('H2 distinct count exceeds the routing domain')
    return grouped


def payload(config, batch, route_rows=None, tp=1):
    h, layers = int(config['hidden_size']), int(config['num_hidden_layers'])
    hq = int(config['num_attention_heads']); hkv = int(config.get('num_key_value_heads', hq))
    d = int(config.get('head_dim', h // hq)); vocab = int(config['vocab_size'])
    experts = int(config.get('num_experts', config.get('num_local_experts', 0)))
    if tp < 1 or hkv % tp:
        raise ValueError('unsupported TP KV partition')
    if experts:
        if tp != 1 or config.get('shared_expert_intermediate_size', 0) or \
                config.get('mlp_only_layers') or config.get('decoder_sparse_step', 1) != 1:
            raise ValueError('H3 MoE payload requires the declared full-MoE TP1 models')
        if route_rows is None or len(route_rows) != layers:
            raise ValueError('MoE bytes require measured per-layer H2 routes')
        inter = int(config.get('moe_intermediate_size', config['intermediate_size']))
        expert_params = 3*h*inter
        dense_params = layers*(h*(hq+2*hkv)*d + h*hq*d + h*experts) + h*vocab
        weights = 2*dense_params + 2*expert_params*sum(float(r['distinct_mean']) for r in route_rows)
        active_params = dense_params + layers*int(config['num_experts_per_tok'])*expert_params
        kv = 4*batch*layers*hkv*d*576
    else:
        dense = dense_payload(config, batch)  # Preserve the R16 logical byte definition.
        weights, kv = dense['gemm_weight_bytes'], dense['kv_read_bytes']
        active_params = weights/2
    flops = 2*batch*active_params + 4*batch*layers*hq*d*576
    # The included QKV/O/MLP/head matrices and KV are rank-sharded. Norm
    # weights, embedding rows and activation are excluded by appendix D.
    return dict(weight_bytes=weights/tp, kv_bytes=kv/tp,
                bytes_step=(weights+kv)/tp, flops_step=flops/tp, layers=layers)


def bound(values, bandwidth_gbps, compute_tflops, tp, nvl_ns):
    memory_ns = values['bytes_step']/positive(bandwidth_gbps, 'bandwidth')
    compute_ns = values['flops_step']/(1000*positive(compute_tflops, 'compute'))
    communication_ns = (2*values['layers']+1)*nvl_ns if tp > 1 else 0
    total = memory_ns+communication_ns if tp > 1 else max(memory_ns, compute_ns)
    return dict(memory_bound_ms=memory_ns/1e6, compute_bound_ms=compute_ns/1e6,
                communication_bound_ms=communication_ns/1e6, bound_ms=total/1e6)


def measurements(directory):
    selected = {}
    for name, default_system in SOURCES:
        path = directory/name
        for line, row in enumerate(read_csv(path), 2):
            if row.get('error') or not row.get('tpot_ms') or row.get('status', '') not in \
                    ('', 'accepted', 'complete') or row.get('protocol_complete', 'true').lower() != 'true':
                continue
            key = (row.get('system') or default_system, row['model'],
                   int(row.get('tp') or 1), int(row['batch']))
            positive(row['tpot_ms'], 'measured TPOT')
            selected[key] = (row, path, line)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'docs/results')
    parser.add_argument('--peaks', type=Path, required=True)
    parser.add_argument('--routing', type=Path, required=True)
    parser.add_argument('--target', type=Path, default=ROOT/'configs/targets/sm_90_h800.json')
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    bw, compute = peaks(args.peaks); routes = routing(args.routing)
    nvl = json.loads(args.target.read_text())['sm90_native_constants']['tp']['nvl_store_ns']
    latencies = {positive(value, 'NVLink latency') for value in nvl.values()}
    if len(latencies) != 1:
        raise ValueError('rank-dependent NVLink latency needs an explicit group mapping')
    nvl_ns = latencies.pop()
    rows = []
    for (system, model, tp, batch), (timing, source, line) in sorted(measurements(args.results).items()):
        config_path = model_dir(model)/'config.json'
        config = json.loads(config_path.read_text())
        values = payload(config, batch, routes.get((model, batch)), tp)
        for basis, bandwidth, throughput in [('measured', bw, compute), ('nominal', 3350., 989.4)]:
            limits = bound(values, bandwidth, throughput, tp, nvl_ns)
            rows.append(dict(system=system, model=model, tp=tp, batch=batch, basis=basis,
                evidence='inferred from verified inputs' if basis == 'measured' else 'inferred from stated nominal peaks',
                bandwidth_gbps=bandwidth, compute_tflops=throughput, **values, **limits,
                tpot_ms=timing['tpot_ms'], bound_over_measured=limits['bound_ms']/float(timing['tpot_ms']),
                timing_source=f'{source}:{line}', timing_source_sha256=sha(source),
                timing_acceptance=timing.get('acceptance', timing.get('status', '')),
                timing_conditions=timing.get('comparable', timing.get('concurrency', '')),
                model_config_sha256=sha(config_path), peaks_sha256=sha(args.peaks),
                routing_sha256=sha(args.routing), target_sha256=sha(args.target)))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.csv.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(args.csv)
    print(json.dumps(dict(rows=len(rows), csv=str(args.csv), scope='LLM; DNN collected separately')))


if __name__ == '__main__':
    main()
