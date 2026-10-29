#!/usr/bin/env python3
"""Aggregate new MoE regular-shape costs without changing dense raw rows."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

CONTRACT = Path(__file__).with_name('moe_regular_cost_measure.py')
spec = importlib.util.spec_from_file_location('regular_cost_contract', CONTRACT)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def checked(ref):
    path = Path(ref['path']); raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('regular cost evidence changed: ' + str(path))
    return path, raw


def record(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def resources(query):
    return dict(mtreg=query['registers'], local_bytes=query['local_bytes'],
        static_smem_bytes=query['static_smem'], resident_ctas=query['resident'])


def validate_sample(row, key, epoch, number, query, reference):
    kind = key['kind']
    contract.validate_key(key, query, reference, epoch['source_sha256'][kind])
    if row['key'] != key or row['round'] != number or row['executor'] != 'mx_static_stage_bench' or row['implementation'] != 'mx5_' + kind + '_v2':
        raise ValueError('regular sample key/round/implementation differs')
    if row['source_sha256'] != epoch['source_sha256'][kind] or row['binary_sha256'] != key['binary_sha256']:
        raise ValueError('regular sample source/binary differs')
    if any(row[name] != value for name, value in resources(query).items()) or row['threads'] != key['threads']:
        raise ValueError('regular sample differs from its actual resource query')
    prefetch = 'prefetch' if kind == 'gemm' else 'prefetch_kind'
    if row[prefetch] != key[prefetch]:
        raise ValueError('regular sample prefetch differs')
    working_set = row.get('active_working_set_bytes', row.get('working_set_bytes'))
    if working_set is None or working_set < 256 * 1024**2 or row['iterations'] <= row['copies']:
        raise ValueError('regular sample rotation is below the prescribed working set')
    for name in ('median_ns', 'p90_ns'):
        if not math.isfinite(row[name]) or row[name] <= 0:
            raise ValueError('regular sample has invalid time')
    if row['p90_ns'] < row['median_ns']:
        raise ValueError('regular p90 is below its median')
    if kind == 'gemm':
        if row['reference_backend'] != 'torch_cpu' or row['reference_precision'] != 'float32' or row['reference_oracle_version'] != 'gemm_cpu_fp32_v1':
            raise ValueError('regular GEMM reference oracle differs')
    elif row['validation'] != dict(context_atol=.006, context_rtol=.016, new_k_atol=.016, new_k_rtol=.016,
            historical_kv_bits=True, repeated_context_bits=True, barrier_and_LA_counts=True):
        raise ValueError('regular attention validation contract differs')


def aggregate(directory, require_complete=True):
    directory = Path(directory).resolve()
    request = json.loads((directory / 'ACTUAL_SOURCE_BOUND_COST_INPUTS_CPU_VALIDATED.json').read_text())
    epoch_path, raw = checked(request['epoch_file']); epoch = json.loads(raw)
    if record(CONTRACT)['sha256'] != epoch['driver']['sha256']:
        raise ValueError('regular aggregation contract differs from the executed wrapper')
    for name in ('driver', 'packing_helper', 'compiler'):
        checked(epoch[name])
    for ref in list(epoch['helpers'].values()) + epoch['inputs']:
        checked(ref)
    queries = json.loads(checked(epoch['queries'])[1])['rows']
    references = json.loads(checked(epoch['references'])[1])['rows']
    mir_ref = next(ref for ref in epoch['inputs'] if ref['path'].endswith('/ACTUAL_795_FITTING_NEW_SHAPE_MIR_CPU_COMPLETE.json'))
    mir_proof = json.loads(checked(mir_ref)[1]); mir = {row['library']['path']: row for row in mir_proof['rows']}
    closure_ref = next(ref for ref in epoch['inputs'] if ref['path'].endswith('/ACTUAL_NEW_SHAPE_GEMM_CPU_RESOURCE_CLOSURE.json'))
    closure = json.loads(checked(closure_ref)[1])['rows']
    table, complete, global_pids = [], [], set()
    for job in request['jobs']:
        proof_path = directory / f"batch{job['index']:03}" / 'ACTUAL_THREE_FRESH_COST_BATCH_COMPLETE.json'
        if not proof_path.exists():
            if require_complete:
                raise ValueError('regular cost batch incomplete: ' + str(job['index']))
            continue
        proof_raw = proof_path.read_bytes()
        if not proof_raw.endswith(b'\n'):
            if require_complete:
                raise ValueError('regular cost proof is being written')
            continue
        proof = json.loads(proof_raw)
        if proof['job'] != job or proof['scope'] != request['epoch_file'] or proof['card'] != 7:
            raise ValueError('regular batch job/scope/card differs')
        keys_path, raw = checked(job['keys']); keys = json.loads(raw)
        if proof['count'] != len(keys) or len(proof['records']) != 3:
            raise ValueError('regular batch is missing keys/rounds')
        samples, pids, numbers, paths = [], set(), set(), set()
        for rr in proof['records']:
            number, pid = rr['round'], rr['PID']
            if number not in (1, 2, 3) or number in numbers or pid in pids or pid in global_pids:
                raise ValueError('regular costs need three fresh processes')
            pids.add(pid); numbers.add(number)
            process_path, raw = checked(rr['process']); process = json.loads(raw); work = process_path.parent
            if process['returncode'] != 0 or json.loads((work / 'guard/guard_result.json').read_text())['code'] != 0 or int((work / 'guard/pgid').read_text()) != pid:
                raise ValueError('regular sample failed its actual guard')
            repo = Path(epoch['packing_helper']['path']).resolve().parents[3]
            expected = [str(repo / 'scripts/mx_gpu_job.sh'), '7', '900', str(work / 'guard'), process['argv'][4],
                epoch['driver']['path'], '--keys', str(keys_path), '--epoch', str(epoch_path), '--out', str(work / 'raw.jsonl'),
                '--round', str(number), '--iterations', '128']
            if process['argv'] != expected or process['physical_card'] != 7:
                raise ValueError('regular actual command differs')
            raw_path, raw = checked(rr['raw'])
            if raw_path in paths:
                raise ValueError('regular rounds reuse a raw file')
            paths.add(raw_path)
            invocation_path, iv_raw = checked(rr['invocation']); iv = json.loads(iv_raw)
            if invocation_path != raw_path.with_suffix('.invocation.json') or iv['PID'] != pid or iv['epoch'] != request['epoch_file'] or iv['keys'] != job['keys'] or iv['raw'] != rr['raw'] or iv['driver'] != epoch['driver'] or iv['helper'] != epoch['helpers'][job['kind']] or iv['round'] != number or not iv['source_bound_cost_samples']:
                raise ValueError('regular same-process driver invocation differs')
            rows = [json.loads(line) for line in raw.decode().splitlines()]
            if len(rows) != len(keys):
                raise ValueError('regular round has missing/extra keys')
            for row, key in zip(rows, keys):
                validate_sample(row, key, epoch, number, queries[key['query_index']], references[key['reference_index']])
            samples.append((number, rows, rr, work))
        global_pids.update(pids)
        for ordinal, key in enumerate(keys):
            kind = key['kind']; query = queries[key['query_index']]; reference = references[key['reference_index']]
            selected = query['key']; base = reference['key']
            for value in (selected, base):
                checked(value['library']); checked(value['metadata'])
            build = json.loads(checked(selected['metadata'])[1]); base_build = json.loads(checked(base['metadata'])[1])
            compiled = mir[key['library']]; checked(compiled['library']); checked(compiled['report'])
            functions = [item['function'] for item in compiled['selected'] if item['function']['function'] == selected['resource']['symbol']]
            if not functions or any(f['calls'] or f['private_accesses'] or not f['rings'] or any(r['generic_loads'] or r['wait_all'] for r in f['rings']) for f in functions):
                raise ValueError('regular selected kernel fails MIR')
            canonical_key = {name: key[name] for name in contract.FIELDS[kind]}
            reference_key = dict(canonical_key); reference_key['prefetch' if kind == 'gemm' else 'prefetch_kind'] = 0
            branch = None
            if kind == 'gemm':
                matches = [case for case in closure if all(case['reference'][name] == key[name] for name in contract.FIELDS[kind] if name != 'prefetch')]
                if len(matches) != 1 or not matches[0]['attempts'][-1]['fit'] or matches[0]['attempts'][-1]['library'] != selected['library']:
                    raise ValueError('regular cost lacks its resource closure')
                branch = dict(closure=closure_ref, case=matches[0])
            provenance = dict(executor='mx_static_stage_bench', implementation='mx5_' + kind + '_v2',
                source_sha256=epoch['source_sha256'][kind], binary_sha256=key['binary_sha256'], base_head=build['base_head'])
            medians = [rows[ordinal]['median_ns'] for _, rows, _, _ in samples]
            p90s = [rows[ordinal]['p90_ns'] for _, rows, _, _ in samples]; median = statistics.median(medians)
            rounds = [dict(round=number, PID=rr['PID'], input=dict(**rr['raw'], line=ordinal + 1),
                invocation=rr['invocation'], process=rr['process'], guard=record(work / 'guard/guard_result.json'), measurement=rows[ordinal])
                for number, rows, rr, work in sorted(samples)]
            table.append(dict(kind=kind, key=canonical_key, median_ns=median, p90_ns=statistics.median(p90s),
                spread=round((max(medians) - min(medians)) / median, 4), processes=3, physical_card=7, rounds=rounds,
                r_next=key['r_next'], r_next_reference=dict(kind=kind, key=reference_key, actual_queries=epoch['references'],
                    query_index=key['reference_index'], resources=resources(reference),
                    provenance=dict(provenance, binary_sha256=base['library']['sha256'], base_head=base_build['base_head'])),
                resources=resources(query), provenance=provenance, resource_branch=branch,
                actual_queries=epoch['queries'], query_index=key['query_index'], selected_MIR=mir_ref))
        complete.append(job['index'])
    if require_complete and (len(table) != request['total_keys'] or len(complete) != len(request['jobs'])):
        raise ValueError('regular cost scope incomplete')
    return dict(schema_version=1, table_kind='mx5_moe_regular_stage_costs', qualified=require_complete,
        qualification_scope='new_moe_regular_shapes_single_source_per_body', complete_scope=require_complete,
        full_model_domain=False, target_defaults_changed=False, collector=record(Path(__file__)), contract=record(CONTRACT),
        scope=request['epoch_file'], source_sha256=epoch['source_sha256'],
        coverage=dict(scope_keys=request['total_keys'], measured_keys=len(table), completed_batches=complete,
            missing_scope_keys=request['total_keys'] - len(table)), rows=table, fresh_processes=len(global_pids))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--audit-prefix', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        p.error('output exists; preserve evidence')
    result = aggregate(a.scope, not a.audit_prefix)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(rows=len(result['rows']), processes=result['fresh_processes'], qualified=result['qualified'], full_model_domain=False)))


if __name__ == '__main__':
    main()
