"""Read-only validation of completed baseline receipts; standard library only."""
import hashlib
import json
import math
from pathlib import Path
import statistics


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


def validate_llm(path, guard):
    path = Path(path)
    value = json.loads(path.read_text())
    if (path.parent/'invalid_measurements.json').exists():
        raise ValueError('invalid measurement receipt coexists with summary')
    if value.get('max_tokens') != 1024 or value.get('prompt_length') != 64:
        raise ValueError('not the final 64/1024 protocol')
    if value.get('valid') is False or value.get('evidence') != 'verified':
        raise ValueError('receipt does not report verified execution')
    package = {'vllm':'vllm', 'sglang':'sglang', 'tensorrt-llm':'tensorrt-llm', 'torch.compile':'transformers'}.get(value.get('system'))
    if not package or not value.get('versions', {}).get('torch') or not value['versions'].get(package):
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
    return value, own
