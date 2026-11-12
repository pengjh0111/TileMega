"""Logical BF16 payload efficiency using frozen TPOT and measured C550 peaks."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_timings():
    rows = {}
    def load(filename):
        with (ROOT / 'docs/results' / filename).open() as stream:
            return list(csv.DictReader(stream))
    def add(row, filename, system='tilemega', tp=1, model=None, field='tpot_ms'):
        model = model or row['model']
        key = model, int(tp), int(row['batch']), system.replace('torchcompile', 'torch.compile')
        value = row.get(field, '')
        rows[key] = dict(model=key[0], tp=key[1], batch=key[2], system=key[3],
            tpot_ms=float(value) if value not in ('', 'N/A', 'NA') else None,
            timing_source='docs/results/' + filename, timing_source_sha256=sha(ROOT/'docs/results'/filename),
            original_record_type=row.get('record_type', ''), timing_remeasured=False)
    for row in load('MX5_llm_performance_snapshot_c550.csv'):
        add(row, 'MX5_llm_performance_snapshot_c550.csv')
    for row in load('MX4_baselines_llm_c550.csv'):
        add(row, 'MX4_baselines_llm_c550.csv', row['system'])
    for row in load('MX5_additional_llm_c550.csv'):
        if row['record_type'].startswith('two_fresh_process_median'):
            add(row, 'MX5_additional_llm_c550.csv', row['system'])
    for row in load('MX5_torchcompile_repair_c550.csv'):
        if row['record_type'] == 'two_fresh_process_median_fullgraph_decode':
            add(row, 'MX5_torchcompile_repair_c550.csv', 'torch.compile')
    for row in load('MX6_final_moe_c550.csv'):
        add(row, 'MX6_final_moe_c550.csv')
    for row in load('MX7_tp_baselines_frozen_c550.csv'):
        add(row, 'MX7_tp_baselines_frozen_c550.csv', row['system'], row['tp'])
    for row in load('MX7_tp_final_c550.csv'):
        add(row, 'MX7_tp_final_c550.csv', tp=row['tp'])
    for row in load('MX7_tp8_final_c550.csv'):
        add(row, 'MX7_tp8_final_c550.csv', row['system'], row['tp'], model='qwen3_1_7b')
    return list(rows.values())


def logical_payload(config, batch, tp=1, active_per_layer=None, isl=64, osl=1024):
    h = config['hidden_size']
    layers = config['num_hidden_layers']
    hq = config['num_attention_heads']
    hkv = config.get('num_key_value_heads', hq)
    d = config.get('head_dim', h // hq)
    experts = config.get('num_local_experts', config.get('num_experts', 0))
    shared = layers * (h*(hq+2*hkv)*d + h*hq*d) + h*config['vocab_size']
    if experts:
        if active_per_layer is None or len(active_per_layer) != layers:
            raise ValueError('MoE payload requires a per-layer active-expert reference')
        if any(not 1 <= n <= min(batch*config['num_experts_per_tok'], experts) for n in active_per_layer):
            raise ValueError('active-expert reference is outside the routing domain')
        intermediate = config.get('moe_intermediate_size', config['intermediate_size'])
        shared += layers*h*experts
        expert_parameters = 3*h*intermediate*sum(active_per_layer)
        parameters = shared + expert_parameters
    else:
        parameters = shared + layers*3*h*config['intermediate_size']
    weight_bytes = 2*parameters/tp
    kv_bytes = 4*batch*layers*hkv*d*(isl+osl/2)/tp
    return dict(weight_bytes_per_rank=weight_bytes, kv_bytes_per_rank=kv_bytes,
        logical_bytes_per_rank=weight_bytes+kv_bytes, average_kv_length=isl+osl/2,
        active_experts_mean=(sum(active_per_layer)/layers if experts else ''),
        payload_definition='BF16 logical weights + distinct active experts + KV; excludes activations, padding, redundant reads and communication')


def hardware_summary(queue):
    groups = collections.defaultdict(list)
    for path in queue.glob('jobs/*/ACCEPTED.json'):
        accepted = json.loads(path.read_text())
        payload = accepted.get('payload') or {}
        if payload.get('item') not in ('hbm_vector', 'hbm_lds', 'bf16_mma'):
            continue
        definition = json.loads(Path(accepted['definition']['path']).read_text())
        for sample in payload['samples']:
            groups[(definition['card'], payload['item'], sample['config'])].append((sample, str(path)))
    rows = []
    for (gpu, item, config), samples in sorted(groups.items()):
        values = [s['value'] for s, _ in samples]
        row = dict(gpu=gpu, item=item, config=config, fresh_processes=len(values),
            evidence='verified', median=statistics.median(values), minimum=min(values), maximum=max(values),
            unit='TFLOPS' if item == 'bf16_mma' else 'GB/s', qualified=len(values) == 5,
            grid=samples[0][0]['grid'], threads=samples[0][0]['threads'],
            dynamic_shared_bytes=samples[0][0]['smem'], sources=json.dumps([p for _,p in samples]))
        rows.append(row)
    complete = len([p for p in queue.glob('jobs/*/ACCEPTED.json') if p.parent.name.startswith(('hbm_', 'bf16_mma_'))]) == 120
    valid = [r for r in rows if r['qualified'] and r['item'].startswith('hbm_')]
    best = max(valid, key=lambda r:r['median']) if complete and valid else None
    return rows, best, complete


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(out, queue, routes):
    peaks, best, complete = hardware_summary(queue)
    write_csv(out/'MX8_hardware_peaks_c550.csv', peaks)
    bw = best['median']*1e9 if best else None
    result = []
    for row in sorted(frozen_timings(), key=lambda r:(r['model'],r['tp'],r['batch'],r['system'])):
        config_path = Path('/root/models') / (row['model']+'_rand') / 'config.json'
        config = json.loads(config_path.read_text())
        experts = config.get('num_local_experts',config.get('num_experts',0))
        active = None
        reference = 'dense logical formula'
        if experts:
            if row['batch'] == 1:
                active = [config['num_experts_per_tok']]*config['num_hidden_layers']
                reference = 'top-k = 8, exactly eight distinct experts per layer at B1'
            else:
                source = routes/f'{row["model"]}_B{row["batch"]}.json'
                if source.exists():
                    capture = json.loads(source.read_text())
                    if capture['positions'] != 1024 or capture['model'] != row['model'] or capture['batch'] != row['batch']:
                        raise ValueError('routing reference scope differs')
                    active = capture['active_experts_per_layer_mean']
                    reference = str(source)+' SHA256 '+sha(source)
        missing = experts and active is None
        payload = logical_payload(config,row['batch'],row['tp'],active) if not missing else dict(
            weight_bytes_per_rank='',kv_bytes_per_rank='',logical_bytes_per_rank='',average_kv_length=576,
            active_experts_mean='',payload_definition='pending exact 1024-position routing reference')
        row.update(model_config_sha256=sha(config_path), category='TP' if row['tp']>1 else ('MoE' if experts else 'dense'),
            **payload, route_reference=reference if not missing else 'missing; do not use a 64-step capture as a 1024-step reference',
            measured_hbm_peak_GBs=best['median'] if best else '', peak_collection_complete=complete,
            bandwidth_efficiency=(payload['logical_bytes_per_rank']/(row['tpot_ms']*1e-3*bw)
                if bw and row['tpot_ms'] and not missing else ''),
            status='pending_routing' if missing else ('pending_hardware' if not bw else ('pass' if row['tpot_ms'] else 'N/A')),
            scope='logical per-rank payload / (frozen TPOT * measured single-card HBM peak); not a DRAM counter')
        result.append(row)
    write_csv(out/'MX8_bandwidth_efficiency_c550.csv',result)
    print(json.dumps(dict(rows=len(result), hardware_complete=complete, measured_hbm_peak_GBs=best['median'] if best else None,
        statuses=dict(collections.Counter(r['status'] for r in result)))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs/mx8/tables')
    parser.add_argument('--queue',type=Path,default=ROOT/'runs/mx8/queue')
    parser.add_argument('--routes',type=Path,default=ROOT/'runs/mx8/routes')
    args=parser.parse_args()
    collect(args.out,args.queue,args.routes)
