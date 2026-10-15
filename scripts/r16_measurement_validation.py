"""Read-only validation of completed baseline receipts; standard library only."""
import hashlib
import json
import math
from pathlib import Path
import statistics


NATIVE_LLM_SYSTEMS = ('TileMega', 'TM-seed', 'TM-H1', 'TM-H1H4', 'TM-sm80', 'TM-noWGMMA')
_HASH_CACHE = {}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = str(path), stat.st_size, stat.st_mtime_ns
    if key not in _HASH_CACHE:
        result = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                result.update(chunk)
        _HASH_CACHE[key] = result.hexdigest()
    return _HASH_CACHE[key]


def native_identity(binary, recorded=None):
    binary = Path(binary)
    identity = json.loads(Path(str(binary)+'.identity.json').read_text())
    if identity.get('schema') != 1 or identity.get('artifact_id') != json_digest({
            key: value for key, value in identity.items() if key != 'artifact_id'}):
        raise ValueError('native artifact identity digest differs')
    if recorded is not None and recorded != identity:
        raise ValueError('recorded native identity differs from its sidecar')
    if identity.get('trace') is not False:
        raise ValueError('instrumented native artifact is ineligible for performance')
    for suffix, key in (('', 'binary_sha256'), ('.cu', 'generated_source_sha256'),
                        ('.plan.json', 'manifest_sha256'), ('.ptxas.log', 'resources_sha256')):
        if file_sha(str(binary)+suffix) != identity.get(key):
            raise ValueError('native artifact sidecar or binary changed: '+suffix)
    if 'build_command_sha256' in identity and file_sha(str(binary)+'.build_command.txt') != identity['build_command_sha256']:
        raise ValueError('native compiler command changed')
    macros = identity.get('complete_macros')
    if macros is not None:
        if macros['compiler_command'] != identity['options']:
            raise ValueError('native macro capture command differs')
        for unit in macros['translation_units']:
            if file_sha(unit['macros_file']) != unit['macros_sha256']:
                raise ValueError('native macro capture changed')
    return identity


def canonical_model(root, config_sha, fallback=None):
    candidates = [directory.name for directory in (Path(root)/'runs/r16/models').glob('*')
                  if (directory/'config.json').is_file() and file_sha(directory/'config.json') == config_sha]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates and fallback is not None:
        return fallback
    raise ValueError('model config SHA does not identify one canonical R16 model')


def native_llm(value, root):
    if root is None:
        raise ValueError('native receipt requires the canonical R16 model registry')
    model = canonical_model(root, value['model_config_sha256'])
    config = value['configuration']
    identities = []
    for index, phase in enumerate(('prefill', 'decode')):
        identity = native_identity(config[phase+'_binary'], config['identities'][phase])
        if identity['binary_sha256'] != config[phase+'_sha256'] or identity['plan'] != config['plan_manifests'][index]:
            raise ValueError('native phase manifest or binary SHA differs')
        identities.append(identity['artifact_id'])
    path = Path(config['checkpoint_manifest'])
    if path.resolve() != (Path(value['model'])/'random_checkpoint.json').resolve():
        raise ValueError('shared checkpoint manifest belongs to a different model directory')
    checkpoint = json.loads(path.read_text())
    payload = {key: checkpoint[key] for key in ('algorithm', 'config_sha256', 'seed', 'aliases', 'tensors')}
    checkpoint_sha = json_digest(payload)
    if (checkpoint['state'] not in ('live', 'closed') or checkpoint['config_sha256'] != value['model_config_sha256'] or
            checkpoint.get('checkpoint_sha256') != checkpoint_sha or config.get('checkpoint_sha256') != checkpoint_sha):
        raise ValueError('shared checkpoint identity differs or supervisor failed')
    return dict(canonical_model=model, native_artifact_ids=','.join(identities),
                source_checkpoint_sha256=checkpoint_sha, checkpoint_identity_shared=True,
                source_checkpoint_label='shared synthetic TileMega/HF checkpoint; pretrained C-1 unavailable')


def native_dnn(value):
    identity = native_identity(value['library_path'], value['identity'])
    if value.get('artifact_id') != identity['artifact_id']:
        raise ValueError('native DNN artifact ID differs')
    execution = value['execution_identity']
    if (execution.get('artifact_id') != identity['artifact_id'] or execution.get('trace') is not False or
            execution.get('execution_id') != json_digest({key: item for key, item in execution.items() if key != 'execution_id'})):
        raise ValueError('native DNN execution identity differs')
    if file_sha(value['numerical_gate_path']) != value['numerical_gate_sha256'] or file_sha(value['input_path']) != value['input_sha256']:
        raise ValueError('native DNN numerical gate or input changed')
    gate = json.loads(Path(value['numerical_gate_path']).read_text())
    if (gate.get('passed') is not True or gate.get('synthetic_weights') or gate.get('artifact_id') != identity['artifact_id'] or
            gate.get('input_tensors_sha256') != value['input_sha256']):
        raise ValueError('native DNN gate does not certify the timed artifact and input')
    if gate.get('archive_sha256') and value.get('exported_program_sha256') != gate['archive_sha256']:
        raise ValueError('native DNN export archive differs from its numerical gate')
    return dict(native_artifact_ids=identity['artifact_id'], source_checkpoint_label='stored export checkpoint certified by native input gate')


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def equal(actual, expected, name):
    positive(actual, name)
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError(f"{name} differs from raw samples/formula")


def ownership(guard, samples):
    acceptance = guard.get('acceptance')
    if acceptance not in ('strict', 'provisional_container_pid_namespace'):
        raise ValueError('unknown guard ownership acceptance')
    unresolved = sorted({pid for sample in samples for pid in sample.get('unresolved_host_pids', [])})
    verified = guard.get('ownership_verified') is True and all(
        sample.get('ownership_verified') is True for sample in samples)
    if acceptance == 'strict' and (not verified or unresolved):
        raise ValueError('strict ownership is unsupported by guard evidence')
    return dict(acceptance=acceptance, ownership_verified=verified, unresolved_host_pids=unresolved)


def telemetry(path):
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError('missing telemetry')
    for sample in samples:
        if sample.get('accepted') is not True:
            raise ValueError('measurement telemetry was rejected')
        positive(sample.get('time'), 'telemetry timestamp')
        data = sample.get('telemetry', {})
        try:
            if int(data['clocks_throttle_reasons.active'], 0) & 0xE8:
                raise ValueError('thermal slowdown or power brake')
        except (KeyError, TypeError):
            raise ValueError('throttle evidence unavailable')
        for name in ('power.draw', 'clocks.sm', 'clocks.mem', 'temperature.gpu'):
            positive(float(data[name]), name)
    return samples


def completed_guard(directory, job):
    if job.get('status') != 'done' or job.get('exit_code') != 0:
        raise ValueError('job did not complete successfully')
    guard = json.loads((directory/'guard_result.json').read_text())
    if guard.get('code') != 0:
        raise ValueError('guard did not accept completed run')
    return guard


def validate_llm(path, guard, root=None):
    path = Path(path)
    value = json.loads(path.read_text())
    if (path.parent/'invalid_measurements.json').exists():
        raise ValueError('invalid measurement receipt coexists with summary')
    if value.get('max_tokens') != 1024 or value.get('prompt_length') != 64:
        raise ValueError('not the final 64/1024 protocol')
    if value.get('valid') is False or value.get('evidence') != 'verified':
        raise ValueError('receipt does not report verified execution')
    package = {'vllm':'vllm', 'sglang':'sglang', 'tensorrt-llm':'tensorrt-llm', 'torch.compile':'transformers'}.get(value.get('system'))
    native = value.get('system') in NATIVE_LLM_SYSTEMS
    if not value.get('versions', {}).get('torch') or (not native and (not package or not value['versions'].get(package))):
        raise ValueError('framework or torch version was not recorded')
    batch = value['batch']
    if type(batch) is not int or batch not in (1, 2, 4, 8, 16):
        raise ValueError('invalid static batch')
    rows, medians = value.get('generation_runs', []), {}
    for count in (1, 1024):
        group = [row for row in rows if row.get('N') == count]
        if (len(group) != 4 or sum(row.get('warmup') is True for row in group) != 1 or
                {row.get('run') for row in group} != {0, 1, 2, 3}):
            raise ValueError('expected one warmup and three timed repetitions')
        timed = []
        for row in group:
            if row.get('warmup') is not (row.get('run') == 0):
                raise ValueError('warmup must precede all timed repetitions')
            elapsed = positive(row.get('wall_seconds'), 'request duration')
            token_path = path.parent/row['tokens_file']
            if token_path.resolve().parent != path.parent.resolve():
                raise ValueError('token file escapes its receipt directory')
            tokens = json.loads(token_path.read_text())
            if (len(tokens) != batch or any(not isinstance(token_row, list) or len(token_row) != count or
                    any(type(token) is not int or token < 0 for token in token_row) for token_row in tokens)):
                raise ValueError('token file has the wrong shape or invalid IDs')
            if row.get('warmup') is False:
                timed.append(elapsed)
            elif row.get('warmup') is not True:
                raise ValueError('warmup flag must be boolean')
        medians[count] = statistics.median(timed)
    if len(rows) != 8:
        raise ValueError('unexpected request measurements')
    equal(value['e2e_seconds'], medians[1024], 'e2e_seconds')
    equal(value['ttft_seconds'], medians[1], 'ttft_seconds')
    equal(value['tpot_seconds'], (medians[1024]-medians[1])/1023, 'tpot_seconds')
    equal(value['output_tokens_per_second'], batch*1024/medians[1024], 'throughput')
    config_file = Path(value['model'])/'config.json'
    if hashlib.sha256(config_file.read_bytes()).hexdigest() != value['model_config_sha256']:
        raise ValueError('model config changed since measurement')
    log = path.parent/'guard.jsonl'
    if Path(value['guard_log']).resolve() != log.resolve():
        raise ValueError('guard log belongs to another measurement')
    samples = telemetry(log)
    labels = {sample['label']: sample for sample in samples}
    if len(labels) != len(samples):
        raise ValueError('duplicate telemetry labels')
    for row in rows:
        for suffix in ('before', 'after'):
            if f"N{row['N']}-run{row['run']}-{suffix}" not in labels:
                raise ValueError('missing request telemetry boundary')
    own = ownership(guard, samples)
    if value.get('ownership_verified') != own['ownership_verified'] or value.get('acceptance') != own['acceptance']:
        raise ValueError('summary ownership differs from its guard evidence')
    if native:
        value['_r16_native'] = native_llm(value, root)
    return value, samples, own


def request_intervals(value, samples):
    labels = {sample['label']: sample for sample in samples}
    result = []
    for row in value['generation_runs']:
        if row['warmup'] or row['N'] != 1024:
            continue
        prefix = f"N1024-run{row['run']}"
        before, after = labels[prefix+'-before']['time'], labels[prefix+'-after']['time']
        duration = row['wall_seconds']
        if after-before < duration-1e-6:
            raise ValueError('wall clock boundaries do not enclose request duration')
        # The actual request lies somewhere within [before,after] and lasts d.
        # Its guaranteed inner interval is [after-d,before+d]; outer intervals
        # alone could overlap only because nvidia-smi took time between requests.
        inner_start, inner_end = after-duration, before+duration
        result.append(dict(run=row['run'], duration=duration, outer_start=before, outer_end=after,
                           inner_start=inner_start, inner_end=max(inner_start, inner_end)))
    return result


def validate_dnn(path, guard):
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get('valid') is False or value.get('evidence') != 'verified' or (path.parent/'invalid_measurements.json').exists():
        raise ValueError('DNN receipt is marked invalid or unverified')
    protocol = value['measurement']
    if protocol.get('warmup') != 50 or protocol.get('iterations') != 200:
        raise ValueError('not the final DNN warmup50/iterations200 protocol')
    values = value.get('samples_ms', [])
    if len(values) != 200:
        raise ValueError('DNN sample count differs')
    for elapsed in values:
        positive(elapsed, 'DNN CUDA event duration')
    equal(value['median_ms'], statistics.median(values), 'DNN median')
    equal(value['p90_ms'], sorted(values)[math.ceil(.9*len(values))-1], 'DNN p90')
    log = path.parent/'guard.jsonl'
    if Path(value['guard_log']).resolve() != log.resolve():
        raise ValueError('DNN guard log belongs to another measurement')
    samples = telemetry(log)
    if not {'before-measurement', 'after-measurement'} <= {sample['label'] for sample in samples}:
        raise ValueError('DNN measurement telemetry missing')
    own = ownership(guard, samples)
    if value.get('system') in ('tilemega', 'TileMega'):
        value['_r16_native'] = native_dnn(value)
    return value, own
