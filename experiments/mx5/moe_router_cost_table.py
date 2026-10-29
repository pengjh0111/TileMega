#!/usr/bin/env python3
"""Aggregate three-process router costs for the recorded dispatch geometry."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def checked(ref):
    path = Path(ref['path'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('router cost evidence changed: ' + str(path))
    return path, raw


def record(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def canonical(key):
    return {**{name: key[name] for name in ('n', 'k', 'b', 'tn', 'wn', 'threads', 'op', 'norm', 'part')},
        'depth': key['d'], 'prefetch': key['pw']}


def resources(query):
    return dict(mtreg=query['registers'], local_bytes=query['local_bytes'],
        static_smem_bytes=query['static_smem'], resident_ctas=query['resident'])


def validate_sample(row, key, epoch, epoch_ref, pid, number, query):
    if query['entry_returncode'] != 0 or not query['fit'] or query['local_bytes'] or query['static_smem'] or query['resident'] != 1 or not 0 < query['registers'] <= (227 if key['threads'] == 576 else 255):
        raise ValueError('router sample has rejected actual device resources')
    if row['key'] != key or row['round'] != number or row['PID'] != pid:
        raise ValueError('router sample key/round/PID differs')
    if not row['cost_protocol'] or not row['source_bound'] or not row['raw_sample_only'] or row['diagnostic_only'] or row['qualified'] or row['final']:
        raise ValueError('router sample is not a formal source-bound raw sample')
    if row['implementation'] != 'mx5_moe_router_v2' or row['executor'] != 'mx_static_stage_bench':
        raise ValueError('router sample implementation differs')
    if row['source_digest'] != epoch['source_digest'] or row['epoch_sha256'] != epoch_ref['sha256'] or row['driver_sha256'] != epoch['driver']['sha256'] or row['legacy_binary_sha256'] != epoch['legacy']['sha256']:
        raise ValueError('router sample source/epoch/driver/legacy differs')
    expected = dict(MTreg=query['registers'], stack=0, static_smem=0, resident=1)
    if row['resources'] != expected:
        raise ValueError('router sample differs from actual device resources')
    if not all(row[field] for field in ('FP32_reference_passed', 'legacy_topk_weights_bitwise', 'dispatch_verified', 'repeat_bitwise')) or row['comparison'] != 'raw_bytes_equal':
        raise ValueError('router sample failed its exact comparison contract')
    if row['copies'] * 128 * 2048 * 2 != row['addressed_rotating_weight_bytes'] or row['addressed_rotating_weight_bytes'] < 256 * 1024**2 or row['iterations'] <= row['copies']:
        raise ValueError('router sample weight rotation differs')
    for name in ('median_ns', 'p90_ns'):
        if not math.isfinite(row[name]) or row[name] <= 0:
            raise ValueError('router sample has invalid time')
    if row['p90_ns'] < row['median_ns'] or not math.isfinite(row['spread_ns']) or row['spread_ns'] < 0:
        raise ValueError('router sample has invalid quantiles/spread')


def aggregate(directory, require_complete=True, *, physical_card=6):
    if type(physical_card) is not int or physical_card not in (7, 6, 5, 4):
        raise ValueError('router collector requires an explicit permitted physical card')
    directory = Path(directory).resolve()
    request = json.loads((directory / 'ACTUAL_SOURCE_BOUND_COST_INPUTS_CPU_VALIDATED.json').read_text())
    epoch_path, raw = checked(request['epoch_file'])
    epoch = json.loads(raw)
    for name in ('driver', 'legacy', 'compiler', 'packing_helper'):
        checked(epoch[name])
    for name, digest in epoch['inputs'].items():
        if hashlib.sha256((Path(epoch['root']) / name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen router source differs: ' + name)
    for ref in epoch['query_records']:
        checked(ref)
    base_ref = next(ref for ref in epoch['query_records'] if ref['path'].endswith('/ACTUAL_ROUTER_FROZEN_BUILD_EPOCH_CPU.json'))
    source_epoch = json.loads(checked(base_ref)[1])
    query_ref = next(ref for ref in epoch['query_records'] if ref['path'].endswith('/ACTUAL_ROUTER_AND_RECOVERED_GROUPED_ZERO_QUERIES.json'))
    actual = json.loads(checked(query_ref)[1])['router']
    queries = {(row['tag'], row['b']): row for row in actual}
    if len(queries) != len(actual) or actual != epoch['eligible_keys']:
        raise ValueError('router query scope differs')
    table, complete, global_pids = [], [], set()
    for job in request['jobs']:
        proof_path = directory / f"batch{job['index']:03}" / 'ACTUAL_THREE_FRESH_COST_BATCH_COMPLETE.json'
        if not proof_path.exists():
            if require_complete:
                raise ValueError('router cost batch incomplete: ' + str(job['index']))
            continue
        proof_raw = proof_path.read_bytes()
        if not proof_raw.endswith(b'\n'):
            if require_complete:
                raise ValueError('router batch proof is being written')
            continue
        proof = json.loads(proof_raw)
        if proof['job'] != job or proof['scope'] != request['epoch_file'] or proof['card'] != physical_card:
            raise ValueError('router batch job/scope/card differs')
        keys_path, raw = checked(job['keys'])
        keys = json.loads(raw)
        if proof['count'] != len(keys) or len(proof['records']) != 3:
            raise ValueError('router batch is missing keys/rounds')
        samples, pids, numbers, raw_paths = [], set(), set(), set()
        for rr in proof['records']:
            number, pid = rr['round'], rr['PID']
            if number not in (1, 2, 3) or number in numbers or pid in pids or pid in global_pids:
                raise ValueError('router batch does not use three fresh processes')
            numbers.add(number); pids.add(pid)
            process_path, raw = checked(rr['process'])
            process = json.loads(raw); work = process_path.parent
            guard = json.loads((work / 'guard/guard_result.json').read_text())
            if process['returncode'] != 0 or guard['code'] != 0 or int((work / 'guard/pgid').read_text()) != pid:
                raise ValueError('router round failed its actual guard')
            repo = Path(epoch['packing_helper']['path']).resolve().parents[3]
            expected_argv = [str(repo / 'scripts/mx_gpu_job.sh'), str(physical_card), '900', str(work / 'guard'),
                process['argv'][4], epoch['driver']['path'], '--keys', str(keys_path), '--epoch', str(epoch_path),
                '--legacy', epoch['legacy']['path'], '--out', str(work / 'raw.jsonl'), '--round', str(number), '--iterations', '128']
            if process['argv'] != expected_argv or process['physical_card'] != physical_card:
                raise ValueError('router actual command differs')
            raw_path, raw = checked(rr['raw'])
            if raw_path in raw_paths:
                raise ValueError('router rounds reuse a raw file')
            raw_paths.add(raw_path)
            rows = [json.loads(line) for line in raw.decode().splitlines()]
            if len(rows) != len(keys):
                raise ValueError('router round has missing/extra keys')
            for row, key in zip(rows, keys):
                validate_sample(row, key, epoch, request['epoch_file'], pid, number, queries[key['tag'], key['b']])
            samples.append((number, rows, rr, work))
        global_pids.update(pids)
        for ordinal, key in enumerate(keys):
            build = json.loads(checked(key['build_record'])[1]); base = json.loads(checked(key['r_next_reference'])[1])
            query = queries[key['tag'], key['b']]; base_query = queries[base['tag'], key['b']]
            kernel = next(v for v in build['kernels'] if v['batch'] == key['b'])
            for field in ('threads', 'tn', 'wn', 'd', 'stream_k', 'pw', 'dispatch_gate_tn', 'dispatch_down_tn'):
                if key[field] != build[field] or (field != 'pw' and base[field] != build[field]):
                    raise ValueError('router cost/build coordinate differs: ' + field)
            if kernel['status'] != 'CPU_resource_fit' or build['pw'] != build['d'] or base['pw'] != 0 or build['binary'] != query['binary'] or base['binary'] != base_query['binary'] or key['r_next'] != base_query['registers']:
                raise ValueError('router build/reference differs from actual query')
            for value in (build, base):
                checked(value['binary'])
                if value['source_digest'] != epoch['source_digest']:
                    raise ValueError('router build source differs')
            report = json.loads(checked(kernel['MIR'])[1]); functions = [f for f in report['functions'] if f"MoeRouterBenchILi{key['b']}EE" in f['function'] and '__device_stub__' not in f['function']]
            if len(functions) != 1 or not functions[0]['rings'] or functions[0]['calls'] or functions[0]['private_accesses'] or any(r['generic_loads'] or r['wait_all'] for r in functions[0]['rings']):
                raise ValueError('router selected kernel fails MIR')
            medians = [rows[ordinal]['median_ns'] for _, rows, _, _ in samples]
            p90s = [rows[ordinal]['p90_ns'] for _, rows, _, _ in samples]
            median = statistics.median(medians); canonical_key = canonical(key)
            provenance = dict(executor='mx_static_stage_bench', implementation='mx5_moe_router_v2',
                source_sha256=epoch['source_digest'], binary_sha256=key['binary_sha256'], base_head=source_epoch['HEAD'])
            reference = dict(kind='moe_router', key=dict(canonical_key, prefetch=0), build=key['r_next_reference'],
                actual_queries=query_ref, resources=resources(base_query),
                provenance=dict(provenance, binary_sha256=base['binary']['sha256']))
            rounds = [dict(round=number, PID=rr['PID'], input=dict(**rr['raw'], line=ordinal + 1),
                process=rr['process'], guard=record(work / 'guard/guard_result.json'), measurement=rows[ordinal])
                for number, rows, rr, work in sorted(samples)]
            table.append(dict(kind='moe_router', key=canonical_key, median_ns=median, p90_ns=statistics.median(p90s),
                spread=round((max(medians) - min(medians)) / median, 4), processes=3, physical_card=physical_card, rounds=rounds,
                r_next=key['r_next'], r_next_reference=reference, resources=resources(query), provenance=provenance,
                resource_build=key['build_record'], dispatch_context=dict(gate_up_tn=key['dispatch_gate_tn'], down_tn=key['dispatch_down_tn'])))
        complete.append(job['index'])
    if require_complete and (len(table) != request['total_keys'] or len(complete) != len(request['jobs'])):
        raise ValueError('router cost scope incomplete')
    return dict(schema_version=1, table_kind='mx5_moe_router_stage_costs', qualified=require_complete,
        qualification_scope=epoch['qualification_scope'], complete_scope=require_complete,
        full_model_domain=False, target_defaults_changed=False, body_bitwise_gate=False,
        collector=record(Path(__file__)), source_digest=epoch['source_digest'], scope=request['epoch_file'],
        dispatch_context=dict(gate_up_tn=epoch['dispatch_gate_tn'], down_tn=epoch['dispatch_down_tn']),
        coverage=dict(scope_keys=request['total_keys'], measured_keys=len(table), completed_batches=complete,
            missing_scope_keys=request['total_keys'] - len(table)), rows=table, fresh_processes=len(global_pids))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--audit-prefix', action='store_true')
    p.add_argument('--card', type=int, choices=(7, 6, 5, 4), default=6)
    a = p.parse_args()
    if a.out.exists():
        p.error('output exists; preserve evidence')
    result = aggregate(a.scope, not a.audit_prefix, physical_card=a.card)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(rows=len(result['rows']), processes=result['fresh_processes'], qualified=result['qualified'], full_model_domain=False)))


if __name__ == '__main__':
    main()
