#!/usr/bin/env python3
"""Aggregate source-bound grouped costs; subset qualification never approves a model."""
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
        raise ValueError('changed evidence: ' + str(path))
    return path, raw


def record(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_sample(row, key, epoch, epoch_ref, pid, number, registers):
    if row['key'] != key or row['round'] != number or row['PID'] != pid:
        raise ValueError('cost sample key/round/PID differs')
    if not row.get('cost_protocol') or not row.get('source_bound') or not row.get('raw_sample_only') or row.get('final'):
        raise ValueError('not a source-bound raw cost sample')
    if row['implementation'] != 'mx5_moe_grouped_v2' or row['executor'] != 'mx_static_stage_bench':
        raise ValueError('cost implementation differs')
    if row['source_digest'] != epoch['source_digest'] or row['epoch_sha256'] != epoch_ref['sha256'] or row['driver_sha256'] != epoch['driver']['sha256']:
        raise ValueError('cost sample source/epoch/driver differs')
    resources = row['resources']
    if resources != dict(MTreg=registers, stack=0, static_smem=0, resident=1):
        raise ValueError('cost sample resource query differs')
    if not row['reference_passed'] or not row['repeat_bitwise']:
        raise ValueError('cost sample failed numerical/determinism checks')
    if row['n_active'] != key['n_active'] or sum(row['row_counts']) != key['b'] * 8 or max(row['row_counts']) - min(row['row_counts']) > 1:
        raise ValueError('grouped routing distribution differs')
    if row['copies'] * key['n_active'] * key['n'] * key['k'] * 2 != row['addressed_rotating_weight_bytes'] or row['addressed_rotating_weight_bytes'] < 256 * 1024**2:
        raise ValueError('cost sample addressed weight rotation is below 256 MiB')
    if row['iterations'] <= row['copies'] or row['iterations'] < 3:
        raise ValueError('cost sample does not visit all copies')
    for field in ('median_ns', 'p90_ns'):
        value = row[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('cost sample has invalid ' + field)
    if row['p90_ns'] < row['median_ns'] or not math.isfinite(row['spread_ns']) or row['spread_ns'] < 0:
        raise ValueError('cost sample has invalid quantiles/spread')


def aggregate(directory, require_complete=True):
    directory = Path(directory).resolve()
    request = json.loads((directory / 'ACTUAL_SOURCE_BOUND_COST_INPUTS_CPU_VALIDATED.json').read_text())
    epoch_path, raw = checked(request['epoch_file'])
    epoch = json.loads(raw)
    checked(epoch['driver'])
    checked(epoch['compiler'])
    checked(epoch['packing_helper'])
    for name, digest in epoch['inputs'].items():
        if hashlib.sha256((Path(epoch['root']) / name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen grouped source changed: ' + name)
    source_ref = next(ref for ref in epoch['query_records'] if ref['path'].endswith('/ACTUAL_FROZEN_RNEXT_BUILD_EPOCH_CPU.json'))
    source_epoch = json.loads(checked(source_ref)[1])
    zero_ref = next(ref for ref in epoch['query_records'] if ref['path'].endswith('/ACTUAL_RNEXT_BASE_ZERO_QUERIES.json'))
    zero_queries = {(row['tag'], row['b']): row for row in json.loads(checked(zero_ref)[1])['rows']}
    table, completed, global_pids = [], [], set()
    queries = {(row['tag'], row['b']): row for row in epoch['eligible_keys']}
    for job in request['jobs']:
        proof_path = directory / f"batch{job['index']:03}" / 'ACTUAL_THREE_FRESH_COST_BATCH_COMPLETE.json'
        if not proof_path.exists():
            if require_complete:
                raise ValueError('cost batch incomplete: ' + str(job['index']))
            continue
        proof_raw = proof_path.read_bytes()
        if not proof_raw.endswith(b'\n'):
            if require_complete:
                raise ValueError('cost batch proof is still being written')
            continue
        proof = json.loads(proof_raw)
        if proof['job'] != job or proof['scope'] != request['epoch_file'] or proof['card'] not in (7, 6):
            raise ValueError('cost batch scope/job/card differs')
        path, raw = checked(job['keys'])
        keys = json.loads(raw)
        if proof['count'] != len(keys) or len(proof['records']) != 3:
            raise ValueError('cost batch is missing keys/rounds')
        samples, paths, pids, numbers = [], set(), set(), set()
        for round_record in proof['records']:
            number, pid = round_record['round'], round_record['PID']
            if number not in (1, 2, 3) or number in numbers or pid in pids or pid in global_pids:
                raise ValueError('cost batch does not use three fresh processes')
            numbers.add(number)
            pids.add(pid)
            process_path, process_raw = checked(round_record['process'])
            process = json.loads(process_raw)
            work = process_path.parent
            guard = json.loads((work / 'guard/guard_result.json').read_text())
            if process['returncode'] != 0 or guard['code'] != 0 or int((work / 'guard/pgid').read_text()) != pid:
                raise ValueError('cost sample did not pass its actual GPU guard')
            repo = Path(epoch['packing_helper']['path']).resolve().parents[3]
            expected_argv = [str(repo / 'scripts/mx_gpu_job.sh'), str(proof['card']), '900',
                str(work / 'guard'), process['argv'][4], epoch['driver']['path'], '--keys', str(path), '--epoch', str(epoch_path),
                '--out', str(work / 'raw.jsonl'), '--round', str(number), '--iterations', '128']
            if process['argv'] != expected_argv or process['physical_card'] != proof['card']:
                raise ValueError('actual GPU cost command differs from the recorded request')
            sample_path, raw = checked(round_record['raw'])
            if sample_path in paths:
                raise ValueError('cost rounds reuse one raw file')
            paths.add(sample_path)
            rows = [json.loads(line) for line in raw.decode().splitlines()]
            if len(rows) != len(keys):
                raise ValueError('cost round has missing/extra keys')
            for row, key in zip(rows, keys):
                query = queries[key['tag'], key['b']]
                validate_sample(row, key, epoch, request['epoch_file'], pid, number, query['registers'])
            samples.append((number, rows, round_record, work))
        global_pids.update(pids)
        for ordinal, key in enumerate(keys):
            build = json.loads(checked(key['build_record'])[1])
            query = queries[key['tag'], key['b']]
            if build['status'] != 'CPU_resource_fit' or build['source_digest'] != epoch['source_digest'] or build['binary']['sha256'] != key['binary_sha256']:
                raise ValueError('grouped cost build is not qualified for its key')
            checked(build['binary'])
            checked(build['MIR'])
            medians = [rows[ordinal]['median_ns'] for _, rows, _, _ in samples]
            p90s = [rows[ordinal]['p90_ns'] for _, rows, _, _ in samples]
            median = statistics.median(medians)
            canonical = {name: key[name] for name in ('n', 'k', 'b', 'n_active', 'tn', 'wn', 'threads', 'op', 'norm', 'part')}
            canonical.update(depth=key['d'], prefetch=key['pw'])
            if key['gate_up']:
                base_ref = key['build_record']
                base_build = build
            else:
                base_sample = next(row for row in build['rows'] if row['b'] == key['b'])
                base_path = Path(key['build_record']['path']).parent.parent / base_sample['tag'] / 'ACTUAL_RNEXT_BASE_BUILD_CPU.json'
                base_ref = record(base_path)
                base_build = json.loads(base_path.read_text())
            base_query = zero_queries[base_build['tag'], key['b']]
            if base_build['pw'] != 0 or base_build['resources']['registers'] != key['r_next'] or base_query['binary'] != base_build['binary']:
                raise ValueError('R_next reference does not match its actual zero-prefetch body')
            checked(base_build['binary'])
            reference_key = dict(canonical, prefetch=0)
            reference = dict(kind='moe_gate_up' if key['gate_up'] else 'moe_down_combine', key=reference_key,
                build=base_ref, actual_queries=zero_ref, resources=dict(mtreg=key['r_next'],
                    local_bytes=base_query['local_bytes'], static_smem_bytes=base_query['static_smem'], resident_ctas=base_query['resident']),
                provenance=dict(executor='mx_static_stage_bench', implementation='mx5_moe_grouped_v2',
                    source_sha256=epoch['source_digest'], binary_sha256=base_build['binary']['sha256'], base_head=source_epoch['HEAD']))
            rounds = []
            for number, rows, rr, work in sorted(samples):
                rounds.append(dict(round=number, PID=rr['PID'], input=dict(**rr['raw'], line=ordinal + 1),
                    process=rr['process'], guard=record(work / 'guard/guard_result.json'), measurement=rows[ordinal]))
            table.append(dict(kind='moe_gate_up' if key['gate_up'] else 'moe_down_combine', key=canonical,
                median_ns=median, p90_ns=statistics.median(p90s), spread=round((max(medians) - min(medians)) / median, 4),
                processes=3, physical_card=proof['card'], rounds=rounds, r_next=key['r_next'], r_next_reference=reference, resource_build=key['build_record'],
                resources=dict(mtreg=query['registers'], local_bytes=0, static_smem_bytes=0, resident_ctas=1),
                provenance=dict(executor='mx_static_stage_bench', implementation='mx5_moe_grouped_v2',
                    source_sha256=epoch['source_digest'], binary_sha256=key['binary_sha256'], base_head=source_epoch['HEAD'])))
        completed.append(job['index'])
    if require_complete and (len(table) != request['total_keys'] or len(completed) != len(request['jobs'])):
        raise ValueError('cost scope is incomplete')
    return dict(schema_version=1, table_kind='mx5_moe_grouped_stage_costs', qualified=require_complete,
        qualification_scope='measured_subset_single_source_epoch_v1', complete_scope=require_complete,
        whole_grouped_domain_complete=False, full_model_domain=False, target_defaults_changed=False,
        body_bitwise_gate=False,
        repeat_check_semantics='Raw repeat_bitwise properties record torch.equal value equality; independent raw-byte body evidence is still required.',
        collector=record(Path(__file__)),
        model='qwen3_30b_a3b_l24', source_digest=epoch['source_digest'], scope=request['epoch_file'],
        coverage=dict(scope_keys=request['total_keys'], measured_keys=len(table), completed_batches=completed,
            missing_scope_keys=request['total_keys'] - len(table), note='Resource-closure/recovered keys require separate evidence; this scope is a subset.'),
        rows=table, fresh_processes=len(global_pids))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--audit-prefix', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        p.error('output exists; retain earlier evidence')
    result = aggregate(a.scope, require_complete=not a.audit_prefix)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(rows=len(result['rows']), processes=result['fresh_processes'], qualified=result['qualified'],
        whole_grouped_domain_complete=False, full_model_domain=False)))


if __name__ == '__main__':
    main()
