#!/usr/bin/env python3
"""Local E0–E2 preparation; preserve the checkpoint's GPU scheduler protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FRAME = HERE.with_name('SERVING_R13')
WORK = Path('/root/r13_sm120_work')
BUILD = Path('/root/tilemega-r13-sm120-build')
TMP = Path('/dev/shm/tilemega-r13-sm120')
PY = '/root/shared-nvme/llm-runtimes/vllm/bin/python'
BASE = '36f17e6ee'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def run(command, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(str(output) + '.command.json', list(map(str, command)))
    with output.open('w') as log:
        code = subprocess.run(list(map(str, command)), cwd=ROOT,
                              stdout=log, stderr=subprocess.STDOUT).returncode
    print(output.name, 'exit', code, flush=True)
    return code


def prepare():
    for directory in (WORK / 'queue', WORK / 'scheduler', HERE / 'raw/E0',
                      TMP / 'models', TMP / 'cache', TMP / 'tmp'):
        directory.mkdir(parents=True, exist_ok=True)
    idle = []
    for i in range(6):
        row = subprocess.check_output([
            'nvidia-smi', '--query-gpu=power.draw,utilization.gpu,memory.used',
            '--format=csv,noheader,nounits'], text=True).strip()
        owners = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'], text=True).strip()
        power, util, used = map(float, row.split(','))
        if owners or util > 5 or used > 1024:
            raise RuntimeError('not idle while establishing local power baseline')
        idle.append(dict(time=time.time(), power_w=power,
                         utilization_pct=util, memory_used_mib=used, owners=owners))
        if i < 5:
            time.sleep(5)
    policy = json.loads((FRAME / 'guard_policy.json').read_text())
    policy['idle_power_w'] = statistics.median(row['power_w'] for row in idle)
    write(HERE / 'guard_policy.json', policy)
    write(HERE / 'raw/E0/idle_power.json', idle)
    # Only local idle power changes; retain the inherited measurement rules.
    measurement = json.loads((HERE.with_name('SERVING_R11') /
                              'ev2/measurement_policy.json').read_text())
    measurement.update(idle_power_w=policy['idle_power_w'], rounds=[],
                       scope='R13 sm120 four-cell checkpoint validation')
    write(HERE / 'measurement_policy.json', measurement)
    for cmd, name in [(['nvidia-smi', '-q'], 'nvidia-smi-q.txt'),
                      (['/usr/local/cuda/bin/nvcc', '--version'], 'nvcc-version.txt')]:
        with (HERE / 'raw/E0' / name).open('w') as out:
            subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=True)
    archive = FRAME / 'raw/sm120_readiness/geometry_inputs.tar.xz'
    manifest = json.loads((FRAME / 'raw/sm120_readiness/inputs_manifest.json').read_text())
    if sha(archive) != manifest['archive_sha256']:
        raise RuntimeError('geometry archive SHA256 mismatch')
    inputs = HERE / 'inputs'
    inputs.mkdir(exist_ok=True)
    subprocess.run(['tar', '-xJf', str(archive), '-C', str(inputs)], check=True)
    for item in manifest['files']:
        if sha(inputs / item['archive_path']) != item['sha256']:
            raise RuntimeError('geometry input mismatch: ' + item['archive_path'])
    fixture_root = Path('/root/shared-nvme/junhuipeng/TileMega/docs/experiments/SEQSCAN/raw/export')
    fixtures = []
    for name in ('gqa2.json', 'mha4.json'):
        source = fixture_root / name
        destination = ROOT / 'docs/experiments/SEQSCAN/raw/export' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cp', '--no-clobber', str(source), str(destination)], check=True)
        fixtures.append(dict(source=str(source), destination=str(destination), sha256=sha(destination)))
    write(HERE / 'raw/E0/unit_fixture_inputs.json', fixtures)
    env = dict(PYTHONPATH=str(ROOT / 'python'), TILEMEGA_BIN=str(BUILD / 'tools/tilemega'),
               TILEMEGA_GPU_LOCK=str(WORK / 'gpu.lock'), TILEMEGA_DEVICE_INDEX='0',
               CUDA_VISIBLE_DEVICES='0', TMPDIR=str(TMP / 'tmp'),
               HF_HOME=str(TMP / 'hf'), HF_HUB_DISABLE_XET='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8',
               PATH='/root/shared-nvme/llm-runtimes/vllm/bin:/usr/local/cuda/bin:' + os.environ['PATH'],
               CUDACXX='/usr/local/cuda/bin/nvcc',
               TILEMEGA_MEASUREMENT_POLICY=str(HERE / 'measurement_policy.json'),
               TORCHINDUCTOR_CACHE_DIR=str(TMP / 'inductor'),
               TRITON_CACHE_DIR=str(TMP / 'triton'),
               FLASHINFER_WORKSPACE_BASE=str(TMP / 'flashinfer'))
    write(HERE / 'launch_env.json', env)
    write(HERE / 'start.json', dict(
        baseline=subprocess.check_output(['git', 'rev-parse', BASE], text=True).strip(),
        upstream_branch='origin/tilemega', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        prompt='/root/Prompt/TileMega_R13_sm120_prompt.md',
        prompt_sha256=sha('/root/Prompt/TileMega_R13_sm120_prompt.md'),
        prompt_origin='User pasted specification; saved locally with LF line endings',
        checkpoint_not_final=True, sm89_phase_d_complete_at_start=False,
        user_overrides=['parallel start before sm89 Phase D completes',
                        'keep installed vLLM 0.29.0, no version switching'],
        inherited_main_protocol_source=str(FRAME),
        geometry_archive_sha256=sha(archive),
        gpu_lock=env['TILEMEGA_GPU_LOCK'], worktree=str(ROOT),
        no_push=True, engineering_budget_seconds=86400,
        storage_note='Fresh tmpfs caches are ephemeral; evidence persists in this worktree. Old R12c caches are not overwritten.',
        known_upstream_failures=['sm120 trigger=0 PTX publication-order check',
                                'nonpaged resource-text identity', 'sm89 Llama B1 B0h C-1']
    ))
    for model, folder in [('llama', 'llama3_2_1b'), ('qwen3', 'qwen3_1_7b')]:
        config = json.loads((ROOT / f'configs/e2e/{model}_r13_sm120.json').read_text())
        config['model']['path'] = str(TMP / 'models' / folder)
        config['device']['cache_dir'] = str(TMP / 'cache' / model)
        config['test']['vllm_python'] = PY
        config['test']['policy_file'] = str(HERE / 'measurement_policy.json')
        config['output']['dir'] = str(HERE / 'raw' / ('E1_' + model))
        write(HERE / f'{model}_r13_sm120.json', config)
    steps = []

    def add(name, command, gpu=False, after=(), after_any=(), timeout=14400):
        steps.append(dict(name=name, command=list(map(str, command)), cwd=str(ROOT), env=env,
                          gpu=gpu, after=list(after), after_any=list(after_any), priority=len(steps),
                          timeout_s=timeout, needs_free_mib=12288,
                          out=str(HERE / 'raw' / name)))

    script = str(HERE / 'bootstrap.py')
    add('E0_core', [sys.executable, script, 'core'], timeout=14400)
    add('E0_test_build', [sys.executable, script, 'test-build'], after=['E0_core'], timeout=21600)
    add('E0_units', [sys.executable, script, 'units'], gpu=True,
        after=['E0_core'], after_any=['E0_test_build'], timeout=10800)
    add('E0_environment', [PY, script, 'environment'], gpu=True,
        after=['E0_core'], after_any=['E0_units'], timeout=900)
    for model in ('llama', 'qwen3'):
        add('E0_model_' + model, [PY, script, 'model', '--model', model],
            after=['E0_core'], timeout=7200)
    for model in ('llama', 'qwen3'):
        add('E1_' + model, [PY, '-m', 'tilemega', 'calibrate', '--config',
                           HERE / f'{model}_r13_sm120.json'], gpu=True,
            after=['E0_environment', 'E0_model_llama', 'E0_model_qwen3'], timeout=21600)
    add('E1_TL2', [PY, script, 'ceiling'], gpu=True,
        after=['E1_llama', 'E1_qwen3'], timeout=14400)
    for suite in ('b', 'c', 'd', 'e', 'f'):
        out = HERE / 'raw' / ('E2a_MB-1' + suite)
        out.mkdir(exist_ok=True)
        add('E2a_MB-1' + suite, [BUILD / 'tilemega-loadbench', '--suite', suite,
                               '--out', out / 'loadbench.json'], gpu=True,
            after=['E1_TL2'], timeout=7200)
    write(HERE / 'queue_e0_e2a.json', steps)
    write(WORK / 'queue/queue_e0_e2a.json', steps)
    print('prepared', len(steps), 'steps; idle median', policy['idle_power_w'], flush=True)


def core():
    stack = Path('/root/shared-nvme/junhuipeng/TileMega')
    folder = HERE / 'raw' / ('E0_core' + os.environ.get('TILEMEGA_E0_BUILD_ATTEMPT', ''))
    command = ['cmake', '-S', ROOT, '-B', BUILD, '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=Release', '-DTILEMEGA_TARGET_ARCH=sm_120',
               '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
               '-DTILEMEGA_BUILD_TESTS=ON', '-DTILEMEGA_BUILD_VERIFY=OFF',
               '-DMLIR_DIR=/root/toolchains/mlir-23a60f15/lib/cmake/mlir',
               '-DLLVM_DIR=/root/toolchains/mlir-23a60f15/lib/cmake/llvm',
               '-DTILEMEGA_LIT_DRIVER=/root/shared-nvme/sxy/cuda-tile/llvm-project/llvm/utils/lit/lit.py']
    for name, folder in [('ISL', 'isl'), ('POLYLIB', 'polylib'), ('BARVINOK', 'barvinok')]:
        command.append(f'-DTILEMEGA_{name}_BUILD_DIR={stack / ("build-" + folder)}')
    command += [f'-DTILEMEGA_ISL_GENERATED_INCLUDE_DIR={stack / "build-isl/include"}',
                f'-DTILEMEGA_ISL_LIBRARY={stack / "build-isl/.libs/libisl.a"}']
    code = run(command, folder / 'configure.log')
    if code:
        return code
    code = run(['cmake', '--build', BUILD, '--target', 'tilemega', 'tilemega-loadbench',
                '--parallel', '3'], folder / 'build.log')
    if code:
        return code
    return run([sys.executable, ROOT / 'python/tilemega/fingerprint.py', '--check',
                BUILD / 'tools/tilemega'], folder / 'fingerprint.log')


def test_build():
    # Keep all individual compile failures visible, but still build their siblings.
    return run(['cmake', '--build', BUILD, '--parallel', '3', '--', '-k', '0'],
               HERE / 'raw/E0_test_build/build.log')


def units():
    codes = []
    codes.append(run(['ctest', '--test-dir', BUILD, '--output-on-failure', '--parallel', '1',
                      '--timeout', '180', '--output-junit', HERE / 'raw/E0_units/ctest.xml'],
                     HERE / 'raw/E0_units/ctest.log'))
    codes.append(run([sys.executable, '-m', 'unittest', 'discover', '-s', str(FRAME),
                      '-p', 'test_*.py'], HERE / 'raw/E0_units/r13_python_tests.log'))
    write(HERE / 'raw/E0_units/status.json', dict(ctest_exit=codes[0], python_exit=codes[1],
          all_tests_attempted=True, validation_passed=not any(codes)))
    return int(any(codes))


def environment():
    import importlib.metadata
    import torch
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'vllm', 'transformers')}
    if versions['vllm'] != '0.29.0':
        raise RuntimeError('installed vLLM changed since preregistration')
    write(HERE / 'env_sm120.json', dict(tilemega_python=sys.executable,
          vllm_python=PY, packages=versions, torch_cuda=torch.version.cuda,
          torch_arch_list=torch.cuda.get_arch_list(), version_switch=False))
    # torch only queries capabilities here, not performance or synchronization.
    properties = torch.cuda.get_device_properties(0)
    write(HERE / 'raw/E0_environment/torch_device.json', dict(
        name=properties.name, major=properties.major, minor=properties.minor,
        total_memory=properties.total_memory, multi_processor_count=properties.multi_processor_count))
    code = run([BUILD / 'tools/tilemega', 'probe', 'device'], HERE / 'raw/E0_environment/device.json')
    if code:
        return code
    query = BUILD / 'sm120_device_query'
    code = run(['/usr/local/cuda/bin/nvcc', '-std=c++17', '-arch=sm_120',
                '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'third_party/cutlass/include'),
                HERE / 'device_query.cu', '-o', query],
               HERE / 'raw/E0_environment/query_build.log')
    if code:
        return code
    return run([query], HERE / 'caps_sm120.json')


def model(name):
    from huggingface_hub import snapshot_download
    prior = json.loads(Path('/root/tilemega-sm120-run/sources_' + name + '.json').read_text())[name]
    destination = TMP / 'models' / ('llama3_2_1b' if name == 'llama' else 'qwen3_1_7b')
    snapshot_download(repo_id=prior['source'], revision=prior['revision'], local_dir=destination,
                      allow_patterns=['*.json', '*.safetensors', 'tokenizer.model', '*.tiktoken', '*.txt'],
                      max_workers=3)
    weights = {path.name: sha(path) for path in sorted(destination.glob('*.safetensors'))}
    record = dict(source=prior['source'], revision=prior['revision'], destination=str(destination),
                  config_sha256=sha(destination / 'config.json'), safetensors_sha256=weights,
                  matches_previous_sm120=weights == prior['safetensors_sha256'],
                  sm89_weight_sha256_status='not present in portable R13 checkpoint; equality not yet verified')
    write(HERE / 'raw' / ('E0_model_' + name) / 'source.json', record)
    if not record['matches_previous_sm120']:
        raise RuntimeError('pinned model weight SHA256 differs from previous local audit')
    return 0


def ceiling():
    sources = []
    for model in ('llama', 'qwen3'):
        doctor = json.loads((HERE / 'raw' / ('E1_' + model) / 'doctor.json').read_text())
        sources.append(Path(doctor['target']))
    code = run([PY, FRAME / 'dram_ceiling.py', '--binary', BUILD / 'tilemega-loadbench',
                '--out', HERE / 'raw/E1_TL2/processes', '--target', sources[0],
                '--target-out', HERE / 'target_sm120.json'], HERE / 'raw/E1_TL2/ceiling.log')
    if code:
        return code
    summary = json.loads((HERE / 'raw/E1_TL2/processes/dram_ceiling.json').read_text())
    # Both model caches use the same native GPU calibration and the TL-2 ceiling.
    sys.path.insert(0, str(FRAME))
    from dram_ceiling import write_target
    for source in sources:
        write_target(source, source, summary['calibration_median_gbps'], summary)
    write(HERE / 'raw/E1_TL2/targets.json', [dict(path=str(p), sha256=sha(p)) for p in sources])
    return 0


def launch():
    existing = WORK / 'scheduler/scheduler.pid'
    if existing.exists():
        pid = int(existing.read_text())
        proc = Path('/proc') / str(pid) / 'cmdline'
        if proc.exists() and b'scheduler.py' in proc.read_bytes():
            raise RuntimeError('existing scheduler is alive; do not launch a duplicate')
    environment = dict(os.environ, **json.loads((HERE / 'launch_env.json').read_text()))
    command = [sys.executable, str(FRAME / 'scheduler.py'), '--queue-dir', str(WORK / 'queue'),
               '--out', str(WORK / 'scheduler'), '--policy', str(HERE / 'guard_policy.json'),
               '--deadline-hours', '96']
    with (WORK / 'scheduler.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(HERE / 'raw/E0/launch.json', dict(pid=process.pid, command=command,
                                          time=time.time(), log=str(WORK / 'scheduler.log')))
    print('scheduler launched', process.pid, flush=True)
    return 0


def recovery():
    import copy
    source = json.loads((HERE / 'queue_e0_e2a.json').read_text())
    rows = copy.deepcopy(source)
    for row in rows:
        row['name'] += '_r1'
        for key in ('after', 'after_any'):
            row[key] = [name + '_r1' for name in row[key]]
        row['out'] += '_r1'
        row['env']['TILEMEGA_E0_BUILD_ATTEMPT'] = '_r1'
    # Publish atomically, without resetting the scheduler or erasing its failures.
    write(HERE / 'queue_e0_recovery.json', rows)
    destination = WORK / 'queue/queue_e0_recovery.json'
    if destination.exists():
        raise RuntimeError('recovery queue already published')
    temporary = destination.with_suffix('.tmp')
    write(temporary, rows)
    temporary.replace(destination)
    print('published', len(rows), 'recovery steps; historical failures unchanged')
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'core', 'test-build', 'units', 'environment', 'model', 'ceiling', 'launch', 'recovery'))
    parser.add_argument('--model', choices=('llama', 'qwen3'))
    args = parser.parse_args()
    return {'prepare': prepare, 'core': core, 'test-build': test_build,
            'units': units, 'environment': environment,
            'model': lambda: model(args.model), 'ceiling': ceiling,
            'launch': launch, 'recovery': recovery}[args.action]() or 0


if __name__ == '__main__':
    raise SystemExit(main())
