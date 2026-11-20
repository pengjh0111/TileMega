"""Validate a completed single-process MPK cohort before admitting its TPOT."""
import argparse
import csv
import re
from pathlib import Path
import subprocess

from a100_baselines import sha
from a100_results import measurement, read
from a800_baselines import RUN
from a800_mpk_preflight import REVISION
from tilemega.bench.scheduler import write_json


def reference(model, checkpoint, prompt):
    with (RUN/'collected/A800_baselines.csv').open() as source:
        rows = [r for r in csv.DictReader(source) if r['model'] == model and r['status'] == 'measured'
                and r['framework'] in ('vllm', 'sglang', 'torchcompile') and
                r['prompt_sha256'] == prompt and r['checkpoint_sha256'] == checkpoint]
    for r in sorted(rows, key=lambda r: -int(r['batch'])):
        for attempt in r['evidence'].split():
            path = RUN/attempt/f"job/measurements/B{r['batch']}/measurements.json"
            data = read(path)
            if (not data or read(RUN/attempt/'guard_result.json').get('acceptance') != 'strict'
                    or data['checkpoint']['checkpoint_sha256'] != checkpoint):
                continue
            run = next(r for r in data['generation_runs'] if r['N'] == 1 and not r['warmup'])
            return path, read(path.parent/run['tokens_file'])
    raise ValueError(f'no guarded independent reference for {model}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort')
    args = parser.parse_args()
    if not args.cohort.startswith('mpk-v') or Path(args.cohort).name != args.cohort:
        parser.error('expected an MPK cohort')
    cohort = RUN/'cohorts'/args.cohort
    state = read(cohort/'baseline/state.json')
    jobs = [j for j in read(cohort/'queue/queue_baselines.json')
            if j.get('engine') == 'mpk' and state.get(j['name'], {}).get('status') == 'done']
    if len(jobs) != 1:
        raise ValueError('expected one completed measurement')
    job = jobs[0]
    step = state[job['name']]
    result = measurement(job, step, read(cohort/'baseline_inputs.json'))
    folder = Path(step['out'])/f"job/measurements/B{job['batch']}"
    raw = read(folder/'measurements.json')
    identity = read(Path(step['out'])/'job/physical-device-identity.json')
    command = job['command']
    policy = Path(command[command.index('--guard-policy') + 1])
    environment = read(policy.parent/'mpk-inputs.json')
    if (policy.parent.parent != cohort/'allocations' or
            sha(policy) != identity['guard_policy_sha256'] or
            environment.get('guard_policy_sha256') != identity['guard_policy_sha256'] or
            environment.get('serving_sha256') != job['harness_sha256'] or
            environment.get('revision') != REVISION):
        raise ValueError('measurement and allocation MPK sources differ')
    cfg = raw['configuration']
    if raw['system'] != 'mpk' or cfg['mirage_revision'] != REVISION or cfg['world_size'] != 1:
        raise ValueError('MPK backend, version or TP mismatch')
    first = None
    artifacts = {}
    for length in (1, 1024):
        runs = [r for r in raw['generation_runs'] if r['N'] == length]
        tokens = read(folder/runs[0]['tokens_file'])
        if any(read(folder/r['tokens_file']) != tokens for r in runs[1:]):
            raise ValueError(f'MPK greedy output diverged between repetitions at N={length}')
        if first is not None and first != [row[:1] for row in tokens]:
            raise ValueError('MPK first token differs between the two compiled lengths')
        first = [row[:1] for row in tokens]
        lib, = (folder/f'kernel_N{length}').glob('mpk_launcher_rank0*.so')
        generated = (lib.parent/'test_rank0.cu').read_text()
        groups = set(re.findall(r'runtime_config\.paged_kv_indptr_buffer\[([^\]]+)\]', generated))
        if groups != {'0'}:
            raise ValueError(f'expected only KV group zero, got {sorted(groups)}')
        listing = subprocess.check_output(['cuobjdump', '--list-elf', str(lib)], text=True, timeout=30)
        if '.sm_80.' not in listing:
            raise ValueError(f'{lib} lacks native SM80 code')
        artifacts[str(length)] = sha(lib)
    path, expected = reference(job['model'], result['checkpoint_sha256'], raw['prompt_ids_sha256'])
    count = min(len(first), len(expected))
    if not count or first[:count] != expected[:count]:
        raise ValueError(f'MPK first tokens differ from {path} over {count} shared prompt rows')
    receipt = dict(measurement_sha256=result['receipt'], checkpoint_sha256=result['checkpoint_sha256'],
                   independent_reference=str(path.relative_to(RUN)), reference_sha256=sha(path),
                   matching_first_tokens=count, sm80_binary_sha256=artifacts,
                   deterministic_repeats=True, environment=environment, tpot_ms=result['tpot_ms'])
    write_json(cohort/'mpk-accepted.json', receipt)
    print(receipt)


if __name__ == '__main__':
    main()
