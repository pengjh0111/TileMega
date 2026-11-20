"""Validate a completed single-process MPK cohort before admitting its TPOT."""
import argparse
import csv
import re
from pathlib import Path
import subprocess

from a100_baselines import inputs, sha
from a100_results import measurement, read
from a800_baselines import RUN
from a800_mpk_preflight import REVISION
from tilemega.bench.scheduler import write_json


def reference(model, checkpoint, prompt, batch):
    with (RUN/'collected/A800_baselines.csv').open() as source:
        rows = [r for r in csv.DictReader(source) if r['model'] == model and r['status'] == 'measured'
                and r['framework'] in ('vllm', 'sglang', 'torchcompile') and
                r['prompt_sha256'] == prompt and r['checkpoint_sha256'] == checkpoint]
    for r in sorted(rows, key=lambda r: (int(r['batch']) != batch, -int(r['batch']))):
        accepted_receipts = set(r['receipt_sha256'].split())
        for attempt in r['evidence'].split():
            path = RUN/attempt/f"job/measurements/B{r['batch']}/measurements.json"
            data = read(path)
            if (not data or sha(path) not in accepted_receipts
                    or data.get('prompt_ids_sha256') != prompt
                    or read(RUN/attempt/'guard_result.json').get('acceptance') != 'strict'
                    or data['checkpoint']['checkpoint_sha256'] != checkpoint):
                continue
            samples = {}
            for length in (1, 1024):
                run = next(r for r in data['generation_runs'] if r['N'] == length and not r['warmup'])
                samples[length] = path.parent/run['tokens_file']
            return path, samples
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
    result = measurement(job, step, read(policy.parent/'baseline_inputs.json'))
    cfg = raw['configuration']
    if raw['system'] != 'mpk' or cfg['mirage_revision'] != REVISION or cfg['world_size'] != 1:
        raise ValueError('MPK backend, version or TP mismatch')
    config, _ = inputs(job['model'])
    vocab_size = read(config/'config.json')['vocab_size']
    first = None
    artifacts = {}
    for length in (1, 1024):
        runs = [r for r in raw['generation_runs'] if r['N'] == length]
        tokens = read(folder/runs[0]['tokens_file'])
        if any(type(token) is not int or not 0 <= token < vocab_size for row in tokens for token in row):
            raise ValueError(f'MPK generated a token outside the configured vocabulary at N={length}')
        if any(read(folder/r['tokens_file']) != tokens for r in runs[1:]):
            raise ValueError(f'MPK greedy output diverged between repetitions at N={length}')
        if first is not None and first != [row[:1] for row in tokens]:
            raise ValueError('MPK first token differs between the two compiled lengths')
        first = [row[:1] for row in tokens]
        lib, = (folder/f'kernel_N{length}').glob('mpk_launcher_rank0*.so')
        generated = (lib.parent/'test_rank0.cu').read_text()
        groups = set(re.findall(r'runtime_config\.paged_kv_indptr_buffer\[([^\]]+)\]', generated))
        if job['model'] == 'llama3_1_8b':
            linear = re.findall(r'linear_kernel<bfloat16,\s*\d+,\s*(\d+),\s*(\d+),\s*(\d+)>', generated)
            gates = {(int(n),int(k)) for n,k,stride in linear if int(stride)==28672}
            if gates != {(256,4096)}:
                raise ValueError(f'MPK Llama 8B gate/up launch geometry mismatch: {gates}')
        if groups != {'0'}:
            raise ValueError(f'expected only KV group zero, got {sorted(groups)}')
        listing = subprocess.check_output(['cuobjdump', '--list-elf', str(lib)], text=True, timeout=30)
        if '.sm_80.' not in listing:
            raise ValueError(f'{lib} lacks native SM80 code')
        artifacts[str(length)] = sha(lib)
    path, samples = reference(job['model'], result['checkpoint_sha256'], raw['prompt_ids_sha256'], job['batch'])
    expected = read(samples[1])
    count = min(len(first), len(expected))
    if not count or first[:count] != expected[:count]:
        raise ValueError(f'MPK first tokens differ from {path} over {count} shared prompt rows')
    expected_full = read(samples[1024])[:count]
    if len(expected_full) != count or any(len(row) != 1024 for row in expected_full):
        raise ValueError('independent 1024-token reference has unexpected dimensions')
    # After the first differing greedy choice the models condition on different
    # prefixes. Keep those later positional matches as descriptive evidence.
    prefix = [next((i for i, (x,y) in enumerate(zip(a,b)) if x!=y), len(a))
              for a,b in zip(tokens[:count], expected_full)]
    equal = [sum(x==y for x,y in zip(a,b)) for a,b in zip(tokens[:count], expected_full)]
    receipt = dict(measurement_sha256=result['receipt'], checkpoint_sha256=result['checkpoint_sha256'],
                   independent_reference=str(path.relative_to(RUN)), reference_sha256=sha(path),
                   reference_n1024_tokens_sha256=sha(samples[1024]),
                   common_prefix_tokens_by_row=prefix, equal_positions_by_row=equal,
                   matching_first_tokens=count, sm80_binary_sha256=artifacts,
                   deterministic_repeats=True, environment=environment, tpot_ms=result['tpot_ms'])
    write_json(cohort/'mpk-accepted.json', receipt)
    print(receipt)


if __name__ == '__main__':
    main()
