"""Build exact coordinates, conservatively deduplicate, and check occupancy."""

import argparse
import concurrent.futures
import os
from pathlib import Path
import time

from calibration_queue import CASE, REPRO, execute
from guarded import ROOT, PYTHON, read, run, sha, write
from make_gemm_probes import identity


TARGET = ROOT / 'configs/targets/sm_90_h800_sq2.json'
TOOLS = CASE.parent / 'build-release/tools'
TOOL = TOOLS / 'tilemega'
PHASE = CASE / 'phase3'
MATERIALIZATIONS = CASE / 'materializations.json'


def compile_command(spec, coord, output):
    return [TOOL, 'compile', spec['export'], output, '--tmexec-config', coord,
            '--serving', 'decode', '--emit', 'serving', '--batch', '1',
            '--runtime-target', TARGET, '--nvcc-split-compile', '4']


def artifact_contract(source, coord, stem=None):
    stem = str(source if stem is None else stem)
    plan = read(stem + '.plan.json')
    assert plan['native_origin'] == 'fixed'
    assert plan['native_dense'] == read(coord)
    assert plan['mode'] == 'L1' and plan['loop'] == 1
    assert plan['grid'] == 132 * plan['native_dense']['residency']
    contract = {k: v for k, v in plan.items() if k not in ('ctg', 'coordinates_sha256', 'native_selection')}
    return {'source_sha256': sha(source), 'full_task_event_ir_sha256': sha(stem + '.tmexec.mlir'),
            'execution_manifest_sha256': identity(contract)}


def generate(spec, coord, directory):
    row_path = directory / 'generation.json'
    if row_path.exists():
        return read(row_path)
    source = directory / 'generated.cu'
    command = compile_command(spec, coord, source)
    previous = directory / 'generate/exit.json'
    code = (read(previous)['returncode'] if previous.exists()
            else run(command, directory / 'generate', dict(os.environ, CUDA_VISIBLE_DEVICES='')))
    if code:
        result = {'status': 'generation_failed', 'coordinates': str(coord), 'log': str(directory / 'generate/run.log')}
    else:
        contract = artifact_contract(source, coord)
        result = {'status': 'generated', 'coordinates': str(coord), 'source': str(source),
                  'realization_id': identity(contract), **contract}
    write(row_path, result)
    return result


def build(spec, generated, directory):
    result_path = directory / 'build.json'
    if result_path.exists():
        return read(result_path)
    binary = directory / 'decode.so'
    coord = Path(generated['coordinates'])
    previous = directory / 'compile/exit.json'
    code = (read(previous)['returncode'] if previous.exists()
            else run(compile_command(spec, coord, binary), directory / 'compile',
                     dict(os.environ, CUDA_VISIBLE_DEVICES='')))
    outcome = Path(str(binary) + '.compile-result.json')
    record = read(outcome) if outcome.exists() else {}
    result = dict(generated, status=record.get('status', 'compiler_error'), binary=str(binary),
                  returncode=code, compile_outcome=str(outcome), build_dir=str(directory))
    if not code:
        assert record['status'] == 'compiled' and record['inputs_unchanged']
        contract = artifact_contract(str(binary) + '.cu', coord, binary)
        assert identity(contract) == generated['realization_id']
        snapshot = read(str(binary) + '.source.json')
        assert snapshot['compiler_sha256'] == sha(TOOL)
        sha_binary = sha(binary)
        assert read(str(binary) + '.identity.json')['binary_sha256'] == sha_binary
        result['binary_sha256'] = sha_binary
    write(result_path, result)
    return result


def materialize(ids, spec, stored, workers):
    records = dict(stored)
    by_implementation = {v['realization_id']: q for q, v in records.items() if v['status'] == 'compiled'}
    aliases = {}
    work = [(q, CASE / 'candidates' / q) for q in ids if q not in records]

    def task(item):
        q, folder = item
        return q, folder, generate(spec, folder / 'coordinates.json', folder / 'original')

    def consume(futures, wait=False):
        ready = list(concurrent.futures.as_completed(futures)) if wait else [f for f in futures if f.done()]
        for future in ready:
            q = futures.pop(future)
            records[q] = future.result()
            write(MATERIALIZATIONS, records)
            print(q, records[q]['status'], flush=True)

    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=3) as generation,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as compilation,
    ):
        pending = {}
        for q, folder, v in generation.map(task, work):
            if v['status'] != 'generated':
                records[q] = v
            elif v['realization_id'] in by_implementation:
                aliases[q] = by_implementation[v['realization_id']]
            else:
                by_implementation[v['realization_id']] = q
                pending[compilation.submit(build, spec, v, folder / 'original')] = q
            consume(pending)
        consume(pending, wait=True)
    for q, reference in aliases.items():
        records[q] = dict(records[reference], implementation_alias_of=reference)
    write(MATERIALIZATIONS, records)
    return records


def query_occupancy(ids, records, pop, phase, pass_number):
    root = CASE / 'occupancy' / phase / f'pass{pass_number}'
    jobs, members = [], {}
    by_rid = {}
    for q in ids:
        row = records[q]
        if row['status'] == 'compiled' and not row.get('occupancy_verified'):
            by_rid.setdefault(row['realization_id'], []).append(q)
    groups = list(by_rid.values())
    for i in range(0, len(groups), 10):
        name = f'group{i // 10:03d}'
        group = groups[i:i + 10]
        manifest = root / f'{name}.json'
        write(manifest, [dict(rank=j, **records[qs[0]]) for j, qs in enumerate(group)])
        command = [PYTHON, str(ROOT / 'scripts/sm90r6_probe_native.py'),
                   '--candidates', str(manifest), '--out', '@ATTEMPT@/probe']
        jobs.append({'id': name, 'command': command})
        members[name] = group
    if not jobs:
        return []
    assert pop['gpu_cards']
    outcomes = execute(jobs, root / 'queue', pop['gpu_cards'], REPRO / 'guard_policy.json')
    changes = []
    for name, outcome in outcomes.items():
        path = Path(outcome['out']).parent / 'probe/result.json'
        resources = read(path)['candidates']
        assert len(resources) == len(members[name])
        for resource in resources:
            qs = members[name][resource['rank']]
            for q in qs:
                row = records[q]
                assert row['binary_sha256'] == resource['binary_sha256']
                row.update(occupancy=resource, occupancy_source=str(path))
                actual = resource['effective_residency']
                if actual < 1:
                    row['status'] = 'resource_rejected'
                elif actual == resource['requested_residency']:
                    row['occupancy_verified'] = True
                else:
                    changes.append((q, actual))
    return changes


def revise(q, row, actual, spec, pass_number):
    directory = CASE / 'candidates' / q / f'capacity-{pass_number}'
    model = read(CASE / 'model_freeze.json')
    output = directory / 'reprice.json'
    command = [TOOLS / 'tilemega-sq2-reprice', TARGET, ROOT, model['cost_table'], spec['export'],
               row['coordinates'], str(actual), output]
    assert run(command, directory / 'reprice') == 0
    corrected = read(output)
    if not corrected['legal']:
        return dict(row, status='resource_rejected', reprice=str(output))
    coord = directory / 'coordinates.json'
    write(coord, corrected['coordinates'])
    generated = generate(spec, coord, directory)
    if generated['status'] != 'generated':
        return dict(generated, parent=row, reprice=str(output))
    result = build(spec, generated, directory)
    return dict(result, parent=row, reprice=str(output),
                realized_pool_ns=corrected['pool_ns'], realized_replay_ns=corrected['replay_ns'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['finalists', 'population'])
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    population = read(CASE / 'population.json')
    assert population['planned_tp_gpu_hours_before_dedup'] <= 24
    for path, expected in population['source_hashes'].items():
        assert sha(path) == expected
    spec = read(REPRO / 'reproduction_inputs.json')
    assert sha(spec['export']) == spec['export_sha256']
    assert sha(TARGET) == read(CASE / 'model_freeze.json')['target_sha256']
    ids = population['S'] if args.phase == 'finalists' else population['P']
    output = CASE / f'{args.phase}-materialized.json'
    if output.exists():
        return
    start = time.time()
    known = read(MATERIALIZATIONS) if MATERIALIZATIONS.exists() else {}
    records = materialize(ids, spec, known, args.workers)
    compiled_at = time.time()
    for revision in range(3):
        changes = query_occupancy(ids, records, population, args.phase, revision)
        write(MATERIALIZATIONS, records)
        if not changes:
            break
        if revision == 2:
            raise RuntimeError('capacity feedback did not converge')
        for q, actual in changes:
            records[q] = revise(q, records[q], actual, spec, revision)
        write(MATERIALIZATIONS, records)
    finished = time.time()
    stats = {q: records[q]['status'] for q in ids}
    result = {'phase': args.phase, 'started': start, 'compiled_at': compiled_at,
              'finished': finished, 'compile_and_generation_wall_seconds': compiled_at - start,
              'occupancy_and_feedback_wall_seconds': finished - compiled_at, 'candidate_status': stats,
              'compiler_sha256': sha(TOOL)}
    if args.phase == 'finalists':
        lookup = {r['id']: r for r in population['all_original_coordinates']}
        valid = [q for q in ids if records[q]['status'] == 'compiled' and records[q]['occupancy_verified']]
        assert valid
        ranked = sorted(valid, key=lambda q: records[q].get('realized_replay_ns', lookup[q]['replay_ns']))
        result.update(predicted=ranked[0], finalists_ranked=ranked)
    write(output, result)
    print(args.phase, len(ids), 'materialized', flush=True)


if __name__ == '__main__':
    main()
