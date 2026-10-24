#!/usr/bin/env python3
"""Validate, stage, run, and collect TileMega-only final E2E measurements."""
import argparse
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write
from run_r16_with_device_identity import require_guard

sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import digest, verify
from tilemega.serving.baseline_common import summarize, validate_prompts
from check_sm90r2_rebuild import encodings

PROTOCOL = dict(isl=64, osl=1024, warmup=1, repeats=3, dtype='bfloat16', sampling='greedy')
MODELS = ('qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_1b', 'llama3_2_3b', 'llama3_1_8b')
TIMING_GPUS = (0, 1, 2, 3, 4, 7)
PCI = {0: '0f', 1: '10', 2: '41', 3: '44', 4: '86', 7: 'b7'}
EMPTY_DIFF = hashlib.sha256(b'').hexdigest()
AMENDMENT = dict(name='user_final_e2e_amendment', scope='TileMega-only E2E timing',
                 c1='not_performed', c2='not_performed', hf='not_performed',
                 external_baselines='not_performed', full_ctest='not_performed')
DEVELOPMENT_CELLS = {'qwen3_0_6b_b1', 'qwen3_0_6b_b16', 'llama3_2_1b_b1',
                     'llama3_2_1b_b16', 'llama3_2_3b_b4', 'llama3_1_8b_b1', 'llama3_1_8b_b16'}


def checked_file(record, base):
    path = Path(record['path'])
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or sha(path) != record['sha256']:
        raise ValueError('manifest file hash mismatch: ' + str(path))
    return path


def prefill_signature(plan):
    fields = ('tile_m', 'tile_n', 'tile_k', 'stages', 'split_k', 'impl')
    return dict(pg=plan['pg'], residency_requested=str(plan['requested_residency']),
                attention_impl=plan['attention_impl'], attention_kv_block=plan['attention_kv_block'],
                gemm_table_sha256=hashlib.sha256(
                    json.dumps(plan['gemms'], sort_keys=True).encode()).hexdigest(),
                geometries=[list(row) for row in sorted({tuple(g[k] for k in fields) for g in plan['gemms']})])


def cpp_json_roundtrip(value):
    # Receipts use Json.cpp's %.10g serializer; the hash still binds the unrounded target bytes.
    if isinstance(value, dict):
        return {k: cpp_json_roundtrip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpp_json_roundtrip(v) for v in value]
    if type(value) in (int, float):
        number = float(value)
        return int(number) if abs(number) <= 9007199254740992 and number.is_integer() else float(format(number, '.10g'))
    return value


def validate_contract(manifest, config, identities, plans, solver, coordinates, rebuild, reference):
    """Check provenance relationships after validating the individual files."""
    if (manifest['schema'] != 1 or manifest['kind'] != 'sm90r2_final_artifacts'
            or manifest['model'] not in MODELS or manifest['batch'] not in (1, 2, 4, 8, 16)
            or manifest['protocol'] != PROTOCOL):
        raise ValueError('final E2E requires a supported dense model and greedy BF16 64/1024 protocol')
    if (config['model'], config['batch']) != (manifest['model'], manifest['batch']):
        raise ValueError('serving config differs from manifest model/batch')
    if config['execution'] != dict(decode='L1', loop=1, prefill='L1'):
        raise ValueError('final E2E requires L1 prefill and native device-loop decode')
    if (solver.get('e3_validated') is not True or solver.get('development_only') is not False
            or solver.get('calibration', {}).get('fit_status') != 'fitted'):
        raise ValueError('development selection is ineligible: final E2E still requires validated E-3 calibration')
    head = manifest['source_head']
    if len(head) != 40 or any(c not in '0123456789abcdef' for c in head):
        raise ValueError('source_head must be a complete committed SHA')
    for role, identity in identities.items():
        source = identity['source']
        if (identity['trace'] or any(k.startswith(('TILEMEGA_TRACE', 'TILEMEGA_PAGE_TRACE',
                'TILEMEGA_SM90_DENSE_BOUNDARY')) and str(v).strip() not in ('0', 'false')
                for k, v in identity['defines'].items())):
            raise ValueError('trace or boundary diagnostic artifact cannot enter final E2E: ' + role)
        if (source['head'] != head or source['worktree_diff_sha256'] != EMPTY_DIFF
                or source['source_digest'] != manifest['source_digest']
                or source['compiler_sha256'] != manifest['compiler']['sha256']):
            raise ValueError('artifact source/compiler does not match the clean final HEAD: ' + role)
        plan = plans[role]
        if (identity['plan'] != plan or plan['batch_lo'] != manifest['batch']
                or plan['batch_hi'] != manifest['batch'] or plan['capacity'] != 1088):
            raise ValueError('artifact plan does not match final workload: ' + role)
    if plans['prefill']['phase'] != 'prefill' or plans['prefill']['seq'] != 64:
        raise ValueError('explicit prefill artifact must implement ISL64 prefill')
    projection = lambda p: {k: v for k, v in p.items() if k not in ('model', 'runtime_target', 'target_spec')}
    if (projection(plans['prefill']) != projection(reference)
            or prefill_signature(reference) != config['coordinates']['prefill']):
        raise ValueError('prefill coordinates differ from the bound reference configuration')
    for role in ('decode', 'selected_candidate'):
        plan = plans[role]
        if (plan.get('native_origin') != 'solver' or plan['phase'] != 'decode'
                or plan['seq'] != 1 or plan['native_dense'] != coordinates):
            raise ValueError('decode must be the exact native solver output: ' + role)
        if identities[role]['target_sha256'] != manifest['target']['sha256']:
            raise ValueError('decode target differs from the bound solver target')
        if plan['export_sha256'] != solver['export_sha256']:
            raise ValueError('decode export differs from the bound solver workload')
    winner = solver['winner']
    if (solver['selection'] != 'measure' or solver['converged'] is not True
            or solver['budget_exhausted'] is not False or type(winner) is not int
            or not 0 <= winner < len(solver['ranked'])
            or solver['ranked'][winner]['rank'] != winner
            or solver['ranked'][winner]['coordinates'] != coordinates
            or solver['target_sha256'] != manifest['target']['sha256']
            or plans['decode']['native_selection'] != solver):
        raise ValueError('solver receipt lacks a completed measured winner with these coordinates')
    selected = identities['selected_candidate']['binary_sha256']
    measurements = [r for r in solver['measurements'] if r['rank'] == winner and r.get('measured')]
    if (len(measurements) != 1 or measurements[0]['status'] != 0
            or measurements[0]['result']['accepted'] is not True
            or measurements[0]['result']['binary_sha256'] != selected):
        raise ValueError('selected candidate lacks its successful measured-selection receipt')
    flags = ('pass', 'same_coordinates', 'same_workload', 'same_source', 'same_instructions',
             'same_instruction_encoding', 'performance_eligible')
    if (any(rebuild.get(k) is not True for k in flags) or rebuild.get('diagnostic') is not False
            or rebuild['reference_sha256'] != selected
            or rebuild['candidate_sha256'] != identities['decode']['binary_sha256']
            or rebuild['generated_source_sha256'] != [identities[r]['generated_source_sha256']
                                                       for r in ('selected_candidate', 'decode')]):
        raise ValueError('final decode lacks a passing bound rebuild comparison')


@lru_cache(maxsize=4)
def committed_source(head, scripts=False):
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'archive', head])
    prefixes = ('include/', 'lib/', 'tools/', 'python/', 'cmake/') + (('scripts/',) if scripts else ())
    suffixes = ('.h', '.hpp', '.cuh', '.cu', '.cpp', '.cc', '.py', '.cmake')
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        return {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                for m in archive if m.isfile() and (m.name.startswith(prefixes)
                or m.name == 'CMakeLists.txt') and (m.name.endswith(suffixes)
                or m.name.endswith('CMakeLists.txt'))}


def validate_e3(evidence, manifest, base):
    if (evidence['schema'] != 1 or evidence['kind'] != 'sm90r2_e3_validation'
            or evidence['source_head'] != manifest['source_head']
            or evidence['target_sha256'] != manifest['target']['sha256']
            or evidence['fit_status'] != 'fitted' or evidence['independent_validation'] is not True
            or evidence['solver_gate_pass'] is not True):
        raise ValueError('E-3 evidence must validate this final source and target independently')
    cells = evidence['cells']
    if len(cells) != 7 or {c['cell'] for c in cells} != DEVELOPMENT_CELLS:
        raise ValueError('E-3 requires all seven development cells')
    for cell in cells:
        errors = cell['class_errors_pct']
        if (set(errors) != {'qkv', 'attention', 'o', 'gate_up', 'down', 'lm_head'}
                or any(not math.isfinite(v) or abs(v) > 15 for v in errors.values())
                or not math.isfinite(cell['step_error_pct']) or abs(cell['step_error_pct']) > 10
                or not math.isfinite(cell['kendall_tau']) or not .7 <= cell['kendall_tau'] <= 1
                or cell['global_solver_top3_contains_fastest'] is not True
                or type(cell['distinct_programs']) is not int or cell['distinct_programs'] < 8):
            raise ValueError('E-3 class/step/ranking gate failed: ' + cell['cell'])
        for key in ('class_evidence', 'step_evidence', 'ranking_evidence'):
            checked_file(cell[key], base)


def validate_manifest(path):
    path = path.resolve(); manifest = read(path); base = path.parent
    files = {key: checked_file(manifest[key], base) for key in
             ('config', 'model_config', 'prompt', 'target', 'compiler', 'solver_receipt',
              'coordinates', 'rebuild', 'prefill_reference_plan', 'guard_policy', 'e3_validation')}
    if files['model_config'].name != 'config.json':
        raise ValueError('model_config must bind the actual config.json loaded by the checkpoint supervisor')
    validate_e3(read(files['e3_validation']), manifest, files['e3_validation'].parent)
    config = read(files['config']); identities, plans, artifacts, sass = {}, {}, {}, {}
    for role in ('prefill', 'decode', 'selected_candidate'):
        artifacts[role] = {k: checked_file(manifest[role][k], base) for k in
                           ('binary', 'identity', 'plan', 'generated_source', 'sass')}
        binary = artifacts[role]['binary']
        for key, suffix in [('identity', '.identity.json'), ('plan', '.plan.json'),
                            ('generated_source', '.cu')]:
            if artifacts[role][key] != Path(str(binary) + suffix):
                raise ValueError('artifact sidecar is not the binary sidecar: ' + role)
        identities[role] = verify(binary)
        plans[role] = read(artifacts[role]['plan'])
        captured = artifacts[role]['sass'].read_text()
        actual = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(binary)],
                                         text=True, timeout=120)
        sass[role] = encodings(actual)
        if sass[role] != encodings(captured):
            raise ValueError('SASS does not belong to the bound binary: ' + role)
    solver = read(files['solver_receipt']); rebuild = read(files['rebuild'])
    validate_contract(manifest, config, identities, plans, solver, read(files['coordinates']),
                      rebuild, read(files['prefill_reference_plan']))
    if (sass['decode'] != sass['selected_candidate']
            or sass['decode'] != rebuild['instruction_encoding_sha256']):
        raise ValueError('selected and rebuilt instruction encodings differ')
    if (subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
            != manifest['source_head']):
        raise ValueError('run from the bound final source HEAD')
    sources = committed_source(manifest['source_head'])
    calibration = read(files['target'])['sm90_dense_decode']
    if (calibration['fit_status'] != 'fitted' or solver['calibration'] != cpp_json_roundtrip(calibration)
            or any(sources.get(p) != h for p, h in calibration['native_headers'].items())):
        raise ValueError('solver calibration/device headers differ from the final committed source')
    for identity in identities.values():
        if identity['source']['source_files'] != sources or digest(sources) != manifest['source_digest']:
            raise ValueError('artifact source snapshot is not the complete committed input set')
    if any(not (ROOT / p).is_file() or sha(ROOT / p) != h for p, h in sources.items()):
        raise ValueError('compiler/runtime inputs differ from the bound source HEAD')
    helpers = committed_source(manifest['source_head'], scripts=True)
    for name in ('run_sm90r2_final_e2e.py', 'run_r16_with_device_identity.py',
                 'sm90d1_common.py', 'check_sm90r2_rebuild.py'):
        if helpers.get('scripts/' + name) != sha(ROOT / 'scripts' / name):
            raise ValueError('E2E helper differs from the bound committed source: ' + name)
    if (config['model_config_sha256'] != manifest['model_config']['sha256']
            or config['prompt_ids_sha256'] != manifest['prompt']['sha256']
            or validate_prompts(read(files['prompt']), manifest['batch']) != 64):
        raise ValueError('model config or frozen ISL64 prompts differ from serving config')
    model = read(files['model_config'])
    if model.get('torch_dtype', model.get('dtype')) != 'bfloat16':
        raise ValueError('model config must declare BF16')
    return dict(manifest=manifest, path=path, files=files, artifacts=artifacts,
                manifest_sha256=sha(path), runner_sha256=sha(__file__))


def validate_gpu(gpu, policy):
    if gpu not in TIMING_GPUS:
        raise ValueError('GPU6 is excluded and GPU5 is functional-only')
    policies = [policy]
    if policy.get('gpu_allowlist_file'):
        policies.append(read(policy['gpu_allowlist_file']))
    for value in policies:
        for key in ('allowed_gpu_indices', 'allowed_timing_gpu_indices'):
            if key in value and gpu not in value[key]:
                raise ValueError('GPU is excluded by the active timing restriction')
    return 0 if gpu < 4 else 1


def command_for(bound, out, fresh):
    m = bound['manifest']; f = bound['files']; artifacts = bound['artifacts']
    return [PYTHON, '-m', 'tilemega.serving.random_checkpoint', '--config', str(f['model_config'].parent),
            '--out', str(out / 'checkpoint'), '--seed', '0', '--max-gib', '32', '--',
            PYTHON, '-m', 'tilemega.serving.r16_run', '--prefill', str(artifacts['prefill']['binary']),
            '--decode', str(artifacts['decode']['binary']), '--prompt-ids', str(f['prompt']),
            '--batch', str(m['batch']), '--max-tokens', '1024', '--warmup', '1', '--repeats', '3',
            '--mode', 'L1', '--decode-loop', '1', '--prefill-mode', 'L1',
            '--round-index', str(fresh), '--out', str(out / 'measurements')]


def process_count(bound):
    return 5 if bound['manifest']['model'] == 'llama3_1_8b' else 3


def require_numa(node):
    ranges = Path(f'/sys/devices/system/node/node{node}/cpulist').read_text().strip().split(',')
    cpus = set()
    for item in ranges:
        ends = [int(n) for n in item.split('-')]
        cpus.update(range(ends[0], ends[-1] + 1))
    if not os.sched_getaffinity(0) <= cpus:
        raise ValueError('timing process is not bound to the GPU NUMA node')


def prepare_queue(bound, gpu, out):
    policy = read(bound['files']['guard_policy']); node = validate_gpu(gpu, policy)
    if out.exists():
        raise FileExistsError('retain the previous queue preparation')
    uuid = policy['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)]
    jobs, previous = [], []
    for fresh in range(process_count(bound)):
        name = f'final-e2e-{bound["manifest"]["model"]}-B{bound["manifest"]["batch"]}-f{fresh}'
        run = [PYTHON, str(Path(__file__).resolve()), 'run', '--manifest', str(bound['path']),
               '--round', str(fresh), '--out', '{attempt_dir}/job']
        wrapper = ['numactl', f'--cpunodebind={node}', PYTHON,
                   str(ROOT / 'scripts/run_r16_with_device_identity.py'),
                   '--guard-policy', str(bound['files']['guard_policy']), '--out', '{attempt_dir}/job',
                   '--expected-uuid', uuid, '--expected-pci', f'0000:{PCI[gpu]}:00.0',
                   '--numa-node', str(node), '--', *run]
        jobs.append(dict(name=name, gpu=True, timing=True, timeout_s=3600, needs_free_mib=32768,
                         cwd=str(ROOT), after=previous, env=dict(CUDA_VISIBLE_DEVICES=str(gpu),
                         TILEMEGA_DEVICE_INDEX=str(gpu), TILEMEGA_NUMA_NODE=str(node),
                         TILEMEGA_GPU_PID_MODE='namespace_limited', PYTHONPATH=str(ROOT / 'python'),
                         OMP_NUM_THREADS='8'), command=wrapper))
        previous = [name]
    out.mkdir(parents=True)
    write(out / 'queue/queue_final_e2e.json', jobs)
    write(out / 'preparation.json', dict(schema=1, kind='sm90r2_final_e2e_preparation',
          manifest_sha256=bound['manifest_sha256'], manifest=str(bound['path']), gpu=gpu,
          jobs=[j['name'] for j in jobs], runner_sha256=bound['runner_sha256'],
          queue_sha256=sha(out / 'queue/queue_final_e2e.json'), active_queue_publication=False,
          scheduler_started=False, gpu_jobs_started=0, user_final_e2e_amendment=AMENDMENT))


def metrics_from_job(bound, out, fresh):
    m = bound['manifest']; path = out / f'measurements/B{m["batch"]}/measurements.json'
    value = read(path); config = value['configuration']
    if (value['system'] != 'TileMega' or value['batch'] != m['batch'] or value['prompt_length'] != 64
            or value['max_tokens'] != 1024 or value['round_index'] != fresh
            or value['prompt_ids_sha256'] != m['prompt']['sha256']
            or value['model_config_sha256'] != m['model_config']['sha256']
            or config['prefill_sha256'] != m['prefill']['binary']['sha256']
            or config['decode_sha256'] != m['decode']['binary']['sha256']
            or (config['mode'], config['decode_loop'], config['prefill_mode']) != ('L1', 1, 'L1')
            or any((out / f'measurements/{name}.json').exists() for name in ('c1', 'c2'))):
        raise ValueError('measurement inputs/execution differ from the final E2E manifest')
    expected = {(n, r, r == 0) for n in (1, 1024) for r in range(4)}
    rows = value['generation_runs']
    if len(rows) != 8 or {(r['N'], r['run'], r['warmup']) for r in rows} != expected:
        raise ValueError('measurement must contain warmup1/repeats3 for N1 and N1024')
    metrics = summarize(rows, m['batch'], 1024)
    if any(value[k] != v for k, v in metrics.items()):
        raise ValueError('reported timing metrics do not match the fresh raw runs')
    tokens = []
    for row in rows:
        token_path = path.parent / row['tokens_file']
        ids = read(token_path)
        if len(ids) != m['batch'] or any(len(r) != row['N'] or any(type(t) is not int or t < 0 for t in r) for r in ids):
            raise ValueError('incomplete generated token table')
        tokens.append(dict(path=str(token_path.resolve()), sha256=sha(token_path),
                           tokens=m['batch'] * row['N'], warmup=row['warmup']))
    return dict(metrics=metrics, measurement=str(path), measurement_sha256=sha(path),
                tokens=tokens, gpu_process_pid=value['pid'], checkpoint=value.get('checkpoint'))


def run(bound, out, fresh, processes=None):
    processes = process_count(bound) if processes is None else processes
    if processes not in (3, 5) or processes < process_count(bound) or not 0 <= fresh < processes:
        raise ValueError('fresh process index is outside the final protocol')
    policy = read(bound['files']['guard_policy']); gpu = int(os.environ['TILEMEGA_DEVICE_INDEX'])
    node = validate_gpu(gpu, policy); require_guard(policy, gpu)
    require_numa(node)
    physical = read(out / 'physical-device-identity.json')
    if (physical['gpu_index'] != gpu or physical['numa_node'] != node
            or physical['cuda_uuid'] != policy['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)].removeprefix('GPU-')
            or physical['cuda_pci_bus_id'].lower() != f'0000:{PCI[gpu]}:00.0'
            or physical['guard_policy_sha256'] != bound['manifest']['guard_policy']['sha256']
            or physical['wrapper_pid'] != os.getppid()
            or physical['wrapper_sha256'] != sha(ROOT / 'scripts/run_r16_with_device_identity.py')
            or os.environ.get('TILEMEGA_TIMING') != '1'
            or os.environ.get('TILEMEGA_NUMA_NODE') != str(node)):
        raise ValueError('physical GPU/NUMA identity differs from the prepared job')
    if any(os.environ.get(k) for k in ('TILEMEGA_RUNTIME_MANIFEST', 'TILEMEGA_M0_BARRIER_DIR')):
        raise ValueError('final E2E cannot inherit a different runtime or a baseline barrier')
    result = out / 'final-e2e-process.json'
    if result.exists() or (out / 'measurements').exists():
        raise FileExistsError('retain prior final E2E process evidence')
    command = command_for(bound, out, fresh)
    evidence = dict(schema=1, kind='sm90r2_final_e2e_process', status='running',
                    manifest_sha256=bound['manifest_sha256'], runner_sha256=bound['runner_sha256'],
                    fresh_process=fresh, started=time.time(), command=command,
                    user_final_e2e_amendment=AMENDMENT)
    write(result, evidence)
    code = subprocess.call(command, cwd=ROOT)
    evidence.update(exit_code=code, ended=time.time(), status='failed' if code else 'awaiting_outer_guard')
    if not code:
        # Revalidate every artifact after the process, before accepting its raw metrics.
        validate_manifest(bound['path'])
        evidence.update(metrics_from_job(bound, out, fresh))
    write(result, evidence)
    return code


def collect(bound, preparation, state_path, out):
    prepared = read(preparation / 'preparation.json'); states = read(state_path)
    if (prepared['manifest_sha256'] != bound['manifest_sha256']
            or prepared['runner_sha256'] != bound['runner_sha256']
            or sha(preparation / 'queue/queue_final_e2e.json') != prepared['queue_sha256']
            or len(prepared['jobs']) != process_count(bound) or out.exists()):
        raise ValueError('preparation/manifest changed or collection output already exists')
    records, pids, uuids = [], set(), set()
    for fresh, name in enumerate(prepared['jobs']):
        state = states[name]
        if state['status'] != 'done' or state['exit_code'] != 0:
            raise ValueError('final E2E scheduler job is not complete: ' + name)
        attempt = Path(state['out']); job = attempt / 'job'; guard = read(attempt / 'guard_result.json')
        evidence = read(job / 'final-e2e-process.json'); physical = read(job / 'physical-device-identity.json')
        wrapper = read(job / 'physical-device-command.json')
        if (guard['code'] != 0 or wrapper['actual_returncode'] != 0
                or evidence['status'] != 'awaiting_outer_guard' or evidence['exit_code'] != 0
                or evidence['manifest_sha256'] != bound['manifest_sha256']
                or evidence['runner_sha256'] != bound['runner_sha256']
                or evidence['fresh_process'] != fresh
                or physical['gpu_index'] != prepared['gpu']):
            raise ValueError('job lacks matching successful scheduler/guard/physical receipts')
        row = metrics_from_job(bound, job, fresh)
        measured = read(row['measurement'])
        if (measured['physical_device_identity'] != physical
                or measured['physical_device_identity_sha256'] != sha(job / 'physical-device-identity.json')
                or evidence['command'] != command_for(bound, job, fresh)
                or evidence['user_final_e2e_amendment'] != AMENDMENT):
            raise ValueError('measurement/command does not carry the bound physical E2E execution')
        if row['gpu_process_pid'] in pids:
            raise ValueError('final E2E requires fresh GPU processes')
        pids.add(row['gpu_process_pid']); uuids.add(physical['cuda_uuid'])
        row.update(guard_sha256=sha(attempt / 'guard_result.json'),
                   physical_identity_sha256=sha(job / 'physical-device-identity.json'),
                   ownership_verified=guard['ownership_verified'])
        records.append(row)
    if len(uuids) != 1:
        raise ValueError('final process cohort changed the physical GPU')
    if not records[0]['checkpoint'] or any(r['checkpoint'] != records[0]['checkpoint'] for r in records):
        raise ValueError('final process cohort changed the deterministic BF16 checkpoint')
    metrics = {key: statistics.median(r['metrics'][key] for r in records) for key in records[0]['metrics']}
    write(out, dict(schema=1, kind='sm90r2_final_e2e', evidence='verified guarded TileMega E2E timing',
          model=bound['manifest']['model'], batch=bound['manifest']['batch'], protocol=PROTOCOL,
          manifest_sha256=bound['manifest_sha256'], source_head=bound['manifest']['source_head'],
          runner_sha256=bound['runner_sha256'], user_final_e2e_amendment=AMENDMENT,
          processes=len(records), physical_uuid=uuids.pop(),
          ownership_verified=all(r['ownership_verified'] for r in records),
          performance_comparison='not performed; single-arm metrics do not establish G-PERF',
          output_tokens_per_generation=1024 * bound['manifest']['batch'], metrics=metrics, records=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', 'queue', 'run', 'collect'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--preparation', type=Path)
    parser.add_argument('--state', type=Path)
    args = parser.parse_args(); bound = validate_manifest(args.manifest)
    if args.action == 'validate':
        print(json.dumps(dict(valid_artifact_contract=True, gpu_jobs_started=0,
                             manifest_sha256=bound['manifest_sha256'], user_final_e2e_amendment=AMENDMENT)))
        return 0
    if args.out is None:
        parser.error('--out is required')
    args.out = args.out.resolve()
    if not args.out.is_relative_to(ROOT / 'runs/sm90r2'):
        parser.error('retain final E2E evidence under runs/sm90r2')
    if args.action == 'queue':
        if args.gpu is None:
            parser.error('queue requires --gpu')
        prepare_queue(bound, args.gpu, args.out)
    elif args.action == 'run':
        return run(bound, args.out, args.round)
    else:
        if args.preparation is None or args.state is None:
            parser.error('collect requires --preparation and --state')
        collect(bound, args.preparation.resolve(), args.state.resolve(), args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
