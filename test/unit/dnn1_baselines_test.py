#!/usr/bin/env python3
"""CPU fixtures only: never CUDA execution or admissible benchmark data."""
import argparse
from collections import Counter
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'scripts/dnn1_baselines.py'
spec = importlib.util.spec_from_file_location('dnn1_baseline_collector', TARGET)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RUNS = ROOT / 'runs/dnn1/cpu-tests'
RUNS.mkdir(parents=True, exist_ok=True)
OUT = Path(tempfile.mkdtemp(prefix='collector-', dir=RUNS))
checks = []


def metrics(model, cosine=1.0):
    return [dict(output=0, cosine_vs_fp32=cosine, top1_agreement_vs_fp32=1.0,
                 psnr_vs_fp32_db=45.0)]


for name in module.MODELS:
    assert module.strict_correctness(name, metrics(name))[0]
    value = metrics(name)
    if name == 'nafnet':
        value[0]['psnr_vs_fp32_db'] = 39.999
    else:
        value[0]['cosine_vs_fp32'] = .9989 if name == 'bert' else .99989
    assert not module.strict_correctness(name, value)[0]
checks.append('model thresholds reject below-gate values')
assert not module.strict_correctness('nafnet', [dict(psnr_vs_fp32_db=50., psnr_min_vs_fp32_db=39.9)])[0]
assert module.strict_correctness('nafnet', [dict(psnr_vs_fp32_db=50., psnr_min_vs_fp32_db=40.)])[0]
assert not module.strict_correctness('bert', [dict(cosine_vs_fp32=1., cosine_min_vs_fp32=.998)])[0]
assert module.strict_correctness('bert', [dict(cosine_vs_fp32=1., cosine_min_vs_fp32=.999)])[0]
assert module.strict_correctness('resnet18', [dict(cosine_min_vs_fp32=.9999, top1_agreement_vs_fp32=1.)])[0]
checks.append('per-image PSNR and per-row cosine minima override passing global metrics')

for arm in module.ARMS:
    for kind in ('compile', 'correctness', 'runtime', 'tune', 'tvm_cuda_build_or_selfcheck'):
        result = dict(final=dict(failure=dict(failure_kind=kind)))
        alternate = module.fallback_arm(arm, result)
        expected = ((arm['system'] == 'torchcompile' and kind == 'compile') or
                    (arm['system'] == 'tvm' and kind in ('compile', 'correctness')))
        assert bool(alternate) == expected
checks.append('fallback only for explicit torch compile or TVM compile/correctness failures')

base = dict(model='resnet18', batch=1, gpu=7, exported_program_sha256='a' * 64,
    source_checkpoint_sha256='b' * 64, prefix=dict(available=False), cell_root=str(OUT / 'cell'),
    python=module.DEFAULT_PYTHON, baselines=str(module.ROOT / 'python/tilemega/dnn/baselines.py'),
    export=str(OUT / 'exports/resnet18'), seed=20261007,
    engine_cache=str(OUT / 'engine-cache'), tvm_root=str(OUT / 'tvm'),
    inductor_cache=str(OUT / 'inductor'), env={}, logical_timeout_s=1,
    protocol=module.PROTOCOL, arms=list(module.ARMS), p_cell_complete=False,
    tvm_selfcheck=str(OUT / 'selfcheck.json'), collector_sha256=module.sha(TARGET))
base['baseline_harness_sha256'] = module.sha(base['baselines'])
module.write(OUT / 'selfcheck.json', dict(cuda_exists=True,
    enabled=dict(USE_CUDA=True, USE_CUBLAS=True, USE_CUDNN=True),
    libinfo=dict(USE_CUDA='ON', USE_CUBLAS='ON', USE_CUDNN='ON')))
module.write(Path(base['cell_root']) / 'onnx-result.json', dict(status='pass'))
prefix = module.prefix_status(dict(base, batch=16))
assert not prefix['available'] and 'known incorrect' in prefix['reason']
prefix = module.prefix_status(dict(base, model='nafnet'))
assert not prefix['available'] and 'sanitizer' in prefix['reason']
checks.append('unsafe native B>1 and NAFNet prefixes are never executed')

native_dir = OUT / 'native-metadata-fixture'
native_dir.mkdir()
for name in ('model.so', 'bridge.json', 'inputs.safetensors'):
    (native_dir / name).write_text('synthetic CPU metadata fixture; never executable')
prefix_fixture = dict(library=str(native_dir / 'model.so'), bridge=str(native_dir / 'bridge.json'),
    inputs=str(native_dir / 'inputs.safetensors'), gate_receipt=str(native_dir / 'gate.json'))
fixture_gate = dict(passed=True, batch=16, model='resnet18',
    library_sha256=module.sha(prefix_fixture['library']), archive_sha256=base['exported_program_sha256'],
    source_checkpoint_sha256=base['source_checkpoint_sha256'], bridge_sha256=module.sha(prefix_fixture['bridge']))
module.write(prefix_fixture['gate_receipt'], fixture_gate)
module.write(str(prefix_fixture['library']) + '.identity.json', dict(trace=False, defines={}))
assert module.prefix_status(dict(base, batch=16, native_prefix=prefix_fixture))['available']
fixture_gate.update(model='nafnet', batch=1)
module.write(prefix_fixture['gate_receipt'], fixture_gate)
assert not module.prefix_status(dict(base, model='nafnet', native_prefix=prefix_fixture))['available']
proofs = {}
for tool in ('memcheck', 'racecheck', 'synccheck', 'initcheck'):
    proof = native_dir / f'{tool}.json'
    module.write(proof, dict(tool=tool, batch=1, passed=True, library_sha256=fixture_gate['library_sha256']))
    proofs[tool] = str(proof)
assert module.prefix_status(dict(base, model='nafnet',
    native_prefix=dict(prefix_fixture, sanitizer_receipts=proofs)))['available']
checks.append('future gated B16 prefix admitted; NAFNet requires matching four-tool sanitizer metadata')

guard_log = OUT / 'clock-fixture.jsonl'
clock_rows = [dict(label=label, gpu_index=7, accepted=True, time=index,
                   telemetry={'clocks.sm': str(clock)})
              for index, (label, clock) in enumerate([('before-initialization', 345),
                  ('before-measurement', 1800), ('after-measurement', 1815)])]
guard_log.write_text(''.join(json.dumps(row) + '\n' for row in clock_rows))
clock_samples = module.measurement_clocks(dict(guard_log=str(guard_log)), 7)
assert clock_samples['clock_start']['sm_clock_mhz'] == 1800
assert clock_samples['clock_end']['sm_clock_mhz'] == 1815
guard_log.write_text(json.dumps(clock_rows[0]) + '\n')
try:
    module.measurement_clocks(dict(guard_log=str(guard_log)), 7)
except ValueError:
    pass
else:
    raise AssertionError('process initialization clock substituted for absent timing samples')
checks.append('CSV clocks use actual measurement boundaries and reject absent samples')

real_selfcheck = dict(cuda_exist=True,
    enabled=dict(USE_CUDA=True, USE_CUBLAS=True, USE_CUDNN=True),
    libinfo=dict(USE_CUDA='/usr/local/cuda', USE_CUBLAS='ON', USE_CUDNN='/workspace/cudnn'))
module.write(OUT / 'selfcheck.json', real_selfcheck)
assert module.validate_tvm_selfcheck(base)['receipt'] == real_selfcheck
library_fixture = OUT / 'selfcheck-library.so'
library_fixture.write_text('CPU metadata fixture; no dynamic library loaded')
selfcheck_with_library = dict(real_selfcheck, libraries=[dict(path=str(library_fixture), sha256=module.sha(library_fixture))])
module.write(OUT / 'selfcheck.json', selfcheck_with_library)
validated = module.validate_tvm_selfcheck(base)
assert validated['path'] == base['tvm_selfcheck']
assert validated['sha256'] == module.sha(base['tvm_selfcheck'])
assert validated['receipt'] == selfcheck_with_library
module.write(OUT / 'selfcheck.json', real_selfcheck)
checks.append('selfcheck provenance remains the receipt path/hash after validating actual library hashes')
for bad in (dict(real_selfcheck, enabled=dict(USE_CUDA=True, USE_CUBLAS=True, USE_CUDNN=False)),
            dict(real_selfcheck, libinfo=dict(USE_CUDA='/usr/local/cuda', USE_CUBLAS='ON', USE_CUDNN='OFF')),
            dict(real_selfcheck, enabled={})):
    module.write(OUT / 'selfcheck.json', bad)
    try:
        module.validate_tvm_selfcheck(base)
    except module.HardStop:
        pass
    else:
        raise AssertionError('inconsistent real-shape TVM selfcheck admitted')
module.write(OUT / 'selfcheck.json', real_selfcheck)
checks.append('real adapter cuda_exist/enabled/path-valued libinfo validated without false DT3')

real_diag = dict(byoc_function_count=3, byoc_functions={'a': 'cublas', 'b': 'cudnn', 'c': 'cublas'},
    byoc_mac_flop_fraction=.75, byoc_total_flop_fraction=None,
    flop_coverage_status='unknown for full graph; static conv/matmul 2MAC subtotal',
    scheduled_tir_function_count=5, cuda_wmma=None, ptx_wmma=None,
    cuda_mma_sync=False, ptx_mma_sync=True, cublas_offload_runtime=True, cudnn_offload_runtime=False,
    libraries=[dict(dynamic_section='NEEDED [libcublas.so.12]\nNEEDED [libcudnn.so.9]', readelf_error=None)],
    tuning=dict(committed_trials=2000, elapsed_s=1499), library_sha256='f' * 64)
normalized = module.normalize_diag(real_diag)
assert normalized['byoc_functions'] == 3 and normalized['byoc_flop_fraction'] == .75
assert normalized['scheduled_tir_functions'] == 5
assert normalized['has_wmma'] is None and normalized['has_mma_sync'] is True
assert normalized['links_cublas'] is True and normalized['links_cudnn'] is True
assert normalized['cublas_offload_runtime'] is True and normalized['cudnn_offload_runtime'] is False
assert normalized['trials'] == 2000 and normalized['tune_wall_seconds'] == 1499
assert normalized['full_graph_flop_fraction'] is None and '2*MAC' in normalized['flop_convention']
assert module.normalize_diag({})['has_wmma'] is None
assert module.normalize_diag(dict(cuda_wmma=False, ptx_wmma=False))['has_wmma'] is False
assert module.normalize_diag(dict(cuda_wmma=False, ptx_wmma=None))['has_wmma'] is None
assert module.normalize_diag({})['links_cublas'] is None
checks.append('real BYOC/TIR/tuning/link diagnostics normalized; unknown instructions remain null; 2MAC/full-graph scope explicit')

arm = module.ARMS[2]
command = module.arm_command(base, dict(arm, freezing=True), 'correctness', 0, OUT / 'job')
assert '--correctness-only' in command and '--freezing' in command
assert command[command.index('--fullgraph') + 1] == 'true'
tvm = module.arm_command(base, module.ARMS[-1], 'measurement', 2, OUT / 'job')
assert tvm[tvm.index('--tvm-max-trials') + 1] == '2000'
assert tvm[tvm.index('--tvm-timeout-s') + 1] == '1800'
checks.append('fixed baseline recipe and correctness-only CLI')

counter = 0
def fake_logical(cell, arm, role, index, destination):
    global counter
    counter += 1
    folder = destination / 'measurements' / f'{role}-{index}' / arm['arm'] / 'attempt1'
    folder.mkdir(parents=True, exist_ok=True)
    data = dict(model=cell['model'], batch=cell['batch'], gpu_index=7,
        system=arm['system'], precision=arm['precision'], pid=10000 + counter,
        median_ms=[1., 3., 2.][index] if role == 'measurement' else None,
        p90_ms=[1.1, 3.1, 2.1][index] if role == 'measurement' else None,
        correctness=metrics(cell['model']), versions={'torch': 'synthetic-fixture'},
        configuration=dict(inputs_sha256='c' * 64, library_sha256='d' * 64),
        process_role=role, measurement=module.PROTOCOL if role == 'measurement' else None)
    failure = None
    if arm['system'] == 'torchcompile' and arm['arm'] == 'torchcompile-max-autotune':
        failure = dict(failure_kind='compile', error='synthetic fullgraph compile failure')
    if arm['arm'] == 'tvm-bf16':
        failure = dict(failure_kind='correctness', error='synthetic TVM BF16 correctness failure')
    if arm['system'] == 'tvm':
        data['configuration']['tvm_diag'] = dict(byoc_functions=4,
            byoc_flop_fraction=.9, scheduled_tir_functions=2, has_wmma=False,
            has_mma_sync=True, links_cublas=True, links_cudnn=True, trials=2000,
            tune_wall_seconds=10, library_sha256='d' * 64)
    measure = folder / 'measurements.json'
    module.write(measure, data)
    final = dict(pid=data['pid'], status='fail' if failure else 'pass',
        clock_start=dict(sm_clock_mhz=1800), clock_end=dict(sm_clock_mhz=1800),
        measurement=str(measure), folder=str(folder), failure=failure)
    return dict(arm=arm, role=role, index=index, status=final['status'], final=final, attempts=[final])

original_logical = module.logical
module.require_session = lambda cell: None
module.logical = fake_logical
original_drain = module.passive_context_drain
module.passive_context_drain = lambda cell: dict(status='CPU_fixture_no_GPU_observation')
cell_path = OUT / 'cell/spec.json'
module.write(cell_path, base)
session_out = OUT / 'raw/cell/attempt1/job'
module.run(argparse.Namespace(cell=cell_path, out=session_out, kind='cell'))
session = module.read(session_out / 'cell-result.json')
assert session['status'] == 'complete'
assert not session['p_cell_complete']
primary = [row['arm'] for row in session['order'] if row.get('phase') == 'primary round']
names = [arm['arm'] for arm in module.ARMS]
assert primary == names + names[::-1] + names
counts = Counter((row['arm']['arm'], row['role']) for row in session['logical_jobs'])
for name in ('torchcompile-max-autotune-fullgraph-false', 'tvm-fp16'):
    assert counts[(name, 'measurement')] == 3
    assert counts[(name, 'correctness')] == 2
assert len(session['logical_jobs']) == 36
assert len({(r['arm']['arm'], r['role'], r['index']) for r in session['logical_jobs']}) == 36
checks.append('entire cell has forward/reverse/forward order plus distinct fallback and 2 correctness processes')

passing_arm = 'tensorrt-bf16'
jobs = [row for row in session['logical_jobs'] if row['arm']['arm'] == passing_arm]
row = module.aggregate_arm(base, session, passing_arm, jobs, True)
assert row['status'] == 'pass'
assert row['median_ms'] == 2 and row['p90_ms'] == 2.1
assert row['process_range_ms'] == 2 and row['process_range_pct'] == 200
assert row['processes'] == 3 and row['correctness_processes'] == 5
assert row['correctness_metrics_identical']
denied = module.aggregate_arm(base, session, passing_arm, jobs, False)
assert denied['status'] == 'N/A' and denied['median_ms'] == '' and denied['processes'] == 0
path = Path(jobs[-1]['final']['measurement'])
original = module.read(path)
different = copy.deepcopy(original)
different['correctness'][0]['cosine_vs_fp32'] = .99999
module.write(path, different)
assert module.aggregate_arm(base, session, passing_arm, jobs, True)['status'] == 'N/A'
v2 = dict(base, correctness_policy='five-independent-gates-v2')
reassessed = module.aggregate_arm(v2, session, passing_arm, jobs, True)
assert reassessed['status'] == 'pass' and reassessed['strict_v1_status'] == 'N/A'
assert not reassessed['correctness_metrics_identical']
different['correctness'][0]['cosine_vs_fp32'] = .9
module.write(path, different)
assert module.aggregate_arm(v2, session, passing_arm, jobs, True)['status'] == 'N/A'
module.write(path, original)
checks.append('median/p90/ranges use 3 timing processes; guard rejection and 5-process mismatch exclude data')

export_cache = Path(base['export'])
export_cache.mkdir(parents=True)
onnx_file = export_cache / 'model_b1_fp32.onnx'
input_file = export_cache / 'inputs_b1.safetensors'
onnx_file.write_text('synthetic ONNX hash fixture; not a model')
input_file.write_text('synthetic input hash fixture; not tensors')
module.write(export_cache / 'onnx_b1.json', dict(batch=1, input_seed=base['seed'],
    exported_program_sha256=base['exported_program_sha256'], path=str(onnx_file),
    sha256=module.sha(onnx_file), native_inputs_file=str(input_file),
    native_inputs_sha256=module.sha(input_file)))
saved_execute = module.execute
def do_not_export(*args, **kwargs):
    raise AssertionError('valid ONNX cache should not run any exporter')
module.execute = do_not_export
module.run_export(base, OUT / 'onnx-reuse')
reused = module.read(OUT / 'onnx-reuse/export-result.json')
assert reused['status'] == 'pass' and reused['reused'] and not reused['attempts']
onnx_file.write_text('changed')
module.run_export(base, OUT / 'onnx-corrupt')
corrupt = module.read(OUT / 'onnx-corrupt/export-result.json')
assert corrupt['status'] == 'fail' and 'hash differs' in corrupt['error']
assert not corrupt['attempts']
module.execute = saved_execute
module.write(Path(base['cell_root']) / 'onnx-result.json', dict(status='pass'))
checks.append('ONNX reuse verifies EP/batch/seed/ONNX/input hashes and never reexports corrupt or valid prior artifacts')

module.write(OUT / 'run/queue-spec.json', dict(cells=[str(cell_path)]))
guard_path = OUT / 'raw/cell/attempt1/guard_result.json'
module.write(guard_path, dict(code=0))
module.write(OUT / 'run/scheduler/state.json', {'dnn1-cell-resnet18_b1':
    dict(status='done', out=str(guard_path.parent))})
module.collect(argparse.Namespace(run_root=OUT / 'run', out=OUT / 'collected'))
with (OUT / 'collected/DNN1_baselines.csv').open() as stream:
    csv_rows = list(csv.DictReader(stream))
assert b'\r\n' not in (OUT / 'collected/DNN1_baselines.csv').read_bytes()
assert b'\r\n' not in (OUT / 'collected/DNN1_tvm_diag.csv').read_bytes()
assert len(csv_rows) == 8 and sum(row['status'] == 'pass' for row in csv_rows) == 6
assert all(row['p_cell_complete'] == 'False' for row in csv_rows)
module.write(guard_path, dict(code=75))
module.collect(argparse.Namespace(run_root=OUT / 'run', out=OUT / 'guard-rejected'))
with (OUT / 'guard-rejected/DNN1_baselines.csv').open() as stream:
    assert all(row['status'] == 'N/A' and not row['median_ms'] for row in csv.DictReader(stream))
checks.append('CSV collection keeps explicit N/A BF16/fullgraph rows and separate reference rows; rejected guards admit none')

module.write(OUT / 'selfcheck.json', dict(cuda_exists=True,
    enabled=dict(USE_CUDA=True, USE_CUBLAS=True, USE_CUDNN=False),
    libinfo=dict(USE_CUDA='ON', USE_CUBLAS='ON', USE_CUDNN='OFF')))
try:
    module.validate_tvm_selfcheck(base)
except module.HardStop:
    pass
else:
    raise AssertionError('cuDNN OFF did not hard-stop')
checks.append('DT-TVM selfcheck OFF hard-stops')

module.write(OUT / 'selfcheck.json', real_selfcheck)
pending_cell = dict(base, tvm_selfcheck=str(OUT / 'not-yet-produced.json'))
try:
    module.validate_tvm_selfcheck(pending_cell)
except module.PreparationPending:
    pass
else:
    raise AssertionError('missing source-build selfcheck was not preparation pending')
pending_path = OUT / 'pending-cell.json'
module.write(pending_path, pending_cell)
try:
    module.run(argparse.Namespace(cell=pending_path, out=OUT / 'pending-cell', kind='cell'))
except module.PreparationPending:
    pass
else:
    raise AssertionError('a cell without selfcheck ran')
assert module.read(OUT / 'pending-cell/cell-result.json')['status'] == 'preparation_pending'
checks.append('missing TVM selfcheck is pending preparation, never evidence of OFF or a GPU measurement')

def all_compile_failures(cell, arm, role, index, destination):
    result = fake_logical(cell, arm, role, index, destination)
    result['status'] = result['final']['status'] = 'fail'
    result['final']['failure'] = dict(failure_kind='compile', error='CPU model-specific compile failure')
    return result
module.logical = all_compile_failures
module.run(argparse.Namespace(cell=cell_path, out=OUT / 'model-compile-failed', kind='cell'))
compile_failed = module.read(OUT / 'model-compile-failed/cell-result.json')
assert compile_failed['status'] == 'complete'
assert all(row['status'] == 'fail' for row in compile_failed['logical_jobs'])
assert any(row['arm']['arm'] == 'tvm-fp16' for row in compile_failed['logical_jobs'])
assert any(row['arm']['arm'].endswith('fullgraph-false') for row in compile_failed['logical_jobs'])
module.logical = fake_logical
checks.append('model compile failures, including both explicit reference arms, finish as N/A without blocking later cells')

fake_now = 0.
def fake_clock():
    return fake_now
def fake_sleep(duration):
    global fake_now
    fake_now += duration
observations = iter([dict(owners={17:16}), dict(owners={}), dict(owners={18:16}),
                     dict(owners={}), dict(owners={})])
drain = original_drain(base, observe=lambda cell: next(observations), clock=fake_clock, sleep=fake_sleep)
assert drain['status'] == 'empty_observed' and len(drain['samples']) == 5
assert drain['elapsed_s'] == 1. and drain['attribution'].startswith('unknown')
fake_now = 0.
drain = original_drain(base, timeout_s=1., observe=lambda cell: dict(owners={23:16}),
                       clock=fake_clock, sleep=fake_sleep)
assert drain['status'] == 'timeout' and drain['elapsed_s'] == 1.
assert all(row['telemetry']['owners'] == {23:16} for row in drain['samples'])
def unavailable(cell):
    raise RuntimeError('CPU fixture NVML unavailable')
drain = original_drain(base, observe=unavailable, clock=fake_clock, sleep=fake_sleep)
assert drain['status'] == 'telemetry_unavailable' and 'NVML unavailable' in drain['error']
checks.append('exit drain is bounded and passive, retains owners without attribution, and requires consecutive empty samples')

original_execute, original_telemetry = module.execute, module.telemetry
module.telemetry = lambda cell: dict(throttle_mask=0, sm_clock_mhz=345)
attempt_counter = 0
def failed_process(command, folder, env, timeout):
    global attempt_counter
    attempt_counter += 1
    folder.mkdir(parents=True, exist_ok=True)
    kind = 'tune' if command[command.index('--system') + 1] == 'tvm' else 'compile'
    module.write(folder / 'failure.json', dict(failure_kind=kind, error='CPU injected ' + kind))
    return dict(returncode=1, pid=90000 + attempt_counter)
module.execute = failed_process
result = original_logical(base, module.ARMS[-1], 'measurement', 0, OUT / 'first-failure-tvm')
assert len(result['attempts']) == 1 and result['final']['failure']['failure_kind'] == 'tune'
assert module.fallback_arm(module.ARMS[-1], result) is None
result = original_logical(base, module.ARMS[2], 'measurement', 0, OUT / 'first-failure-torch')
assert len(result['attempts']) == 1 and module.fallback_arm(module.ARMS[2], result)['fullgraph'] is False
result = original_logical(base, module.ARMS[0], 'measurement', 0, OUT / 'max-retry-trt')
assert len(result['attempts']) == 2
module.execute, module.telemetry = original_execute, original_telemetry
checks.append('TVM tune/runtime classification cannot change on cache retry; explicit fullgraph compile fails reference directly; other retries capped at one')

export = OUT / 'queue-export/resnet18'
(export / 'checkpoint').mkdir(parents=True)
(export / 'exported_program.pt2').write_text('synthetic CPU fixture; not a real exported model')
(export / 'checkpoint/model.safetensors').write_text('synthetic CPU fixture; not real weights')
module.write(export / 'manifest.json', dict(model='resnet18', timing_eligible=True,
    artifacts={'exported_program.pt2': module.sha(export / 'exported_program.pt2')},
    checkpoint={'sha256': module.sha(export / 'checkpoint/model.safetensors')}))
module.write(OUT / 'queue-input.json', dict(gpu=7, exports_root=str(export.parent),
    tvm_selfcheck=str(OUT / 'selfcheck.json'),
    cells=[dict(model='resnet18', batch=1), dict(model='resnet18', batch=16)]))
module.queue(argparse.Namespace(spec=OUT / 'queue-input.json', out=OUT / 'queue-output'))
queued = module.read(OUT / 'queue-output/queue/queue_dnn1_baselines.json')
assert len(queued) == 4
assert queued[0]['gpu'] is False and queued[0]['command'][0] == 'flock'
assert queued[1]['gpu'] is True and queued[1]['timing'] is True
assert queued[2]['after'] == [queued[1]['name']]
assert queued[3]['after'] == [queued[2]['name']]
for path in module.read(OUT / 'queue-output/queue-spec.json')['cells']:
    queued_cell = module.read(path)
    assert queued_cell['python'] == module.DEFAULT_PYTHON
    assert queued_cell['env']['PYTHONPATH'].split(':')[:2] == [str(module.ROOT / 'python'), module.DEFAULT_EXTRA_PATH]
checks.append('queue gives each cell one outer guard; CPU export precedes it; all cells chained; venv symlink retained')

module.preflight(argparse.Namespace(run_root=OUT / 'queue-output'))
assert module.read(OUT / 'queue-output/preflight.json')['status'] == 'ready'
first_cell = Path(module.read(OUT / 'queue-output/queue-spec.json')['cells'][0])
changed = module.read(first_cell)
changed['recipe_sources'][str(OUT / 'adapter-not-final.py')] = None
module.write(first_cell, changed)
try:
    module.preflight(argparse.Namespace(run_root=OUT / 'queue-output'))
except module.PreparationPending:
    pass
else:
    raise AssertionError('unfinished baseline source admitted by preflight')
assert module.read(OUT / 'queue-output/preflight.json')['status'] == 'preparation_pending'
module.queue(argparse.Namespace(spec=OUT / 'queue-input.json', out=OUT / 'queue-output', refresh_prepared=True))
assert len(list((OUT / 'queue-output/prepared_revisions').iterdir())) == 1
(OUT / 'queue-output/scheduler').mkdir()
try:
    module.queue(argparse.Namespace(spec=OUT / 'queue-input.json', out=OUT / 'queue-output', refresh_prepared=True))
except ValueError:
    pass
else:
    raise AssertionError('started scheduler evidence could be overwritten by queue refresh')
checks.append('CPU preflight enforces final sources/selfcheck; refresh archives only unstarted preparations')

module.write(OUT / 'validation-summary.json', dict(
    evidence='synthetic CPU fixtures only; no CUDA, model compilation or performance data',
    passed=True, checks=checks, collector_sha256=module.sha(TARGET)))
print(json.dumps(dict(passed=True, checks=len(checks), out=str(OUT)), indent=2))
