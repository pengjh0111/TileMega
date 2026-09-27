#!/usr/bin/env python3
"""Archive a completed, predeclared Llama B=1/16 EV-2 run without altering decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def load(path: Path):
    return json.loads(path.read_text())


def final_power(records: list[dict], label: str) -> dict:
    matches = [row for row in records if row.get('label') == label]
    accepted = [row for row in matches if row.get('power_accepted') is True]
    if not accepted:
        raise ValueError(f'no accepted power observation for {label}')
    return accepted[-1]


def exclusive(records: list[dict], label: str) -> bool:
    matches = [row for row in records if row.get('label') == label]
    if not matches:
        raise ValueError(f'no process observation for {label}')
    return matches[-1].get('exclusive') is True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--evidence', type=Path,
                        default=Path(__file__).resolve().parent / 'ev2')
    args = parser.parse_args()
    run, evidence = args.run.resolve(), args.evidence.resolve()
    policy_path = evidence / 'measurement_policy.json'
    policy = load(policy_path)
    if policy['rounds']:
        raise ValueError('predeclared policy already has decisions; archive only once')
    if load(run / 'config.json')['workload']['batch'] != [1, 16]:
        raise ValueError('run does not match the narrowed B=1/16 scope')
    raw = evidence / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for batch in (1, 16):
        for arm in ('tilemega', 'vllm'):
            folder = run / f'B{batch}' / arm
            if arm == 'vllm':
                folder /= f'B{batch}'
            measurements = load(folder / 'measurements.json')
            records = [json.loads(line) for line in
                       (folder / 'guard.jsonl').read_text().splitlines()]
            observations = []
            for item in measurements['runs']:
                if item['N'] != 1024 or item['warmup']:
                    continue
                n = item['run']
                after = [record for record in records if
                         record.get('label', '').startswith(f'N1024-run{n}-attempt') and
                         record.get('label', '').endswith('-after') and
                         record.get('power_accepted') is True]
                if not after:
                    raise ValueError(f'no accepted timing attempt: B{batch} {arm} run {n}')
                attempt = after[-1]['label'].split('-')[2]
                prefix = f'N1024-run{n}-{attempt}'
                before = final_power(records, prefix + '-before')
                finish = final_power(records, prefix + '-after')
                is_exclusive = exclusive(records, f'N1024-run{n}-before') and \
                    exclusive(records, f'N1024-run{n}-after')
                threshold = float(finish['power_threshold_w'])
                power = max(float(before['power_w']), float(finish['power_w']))
                accepted = is_exclusive and power <= threshold
                observations.append(dict(run=n, attempt=attempt, power_w=power,
                                         threshold_w=threshold,
                                         exclusive=is_exclusive, accepted=accepted))
            if len(observations) != 3 or not all(x['accepted'] for x in observations):
                raise ValueError(f'incomplete clean timed arm: B{batch} {arm}')
            policy['rounds'].append(dict(model='llama', batch=batch, arm=arm,
                                         observations=observations))
            shutil.copy2(folder / 'guard.jsonl', raw / f'llama_B{batch}_{arm}_guard.jsonl')
            shutil.copy2(folder / 'measurements.json', raw / f'llama_B{batch}_{arm}.json')
        for name in ('hf_tilemega.json', 'hf_vllm.json'):
            shutil.copy2(run / f'B{batch}' / name, raw / f'llama_B{batch}_{name}')
    plans = load(run / 'plans.json')
    sass = []
    for batch in (1, 16):
        for phase in ('prefill', 'decode'):
            binary = Path(plans[str(batch)][phase])
            audit = load(binary.parent / 'sass.json')
            sass.append(dict(batch=batch, phase=phase, binary=str(binary),
                             fp64_total=audit['fp64_total'], audit=audit))
    all_zero = all(row['fp64_total'] == 0 for row in sass)
    (evidence / 'sass_audit.json').write_text(json.dumps(
        dict(all_fp64_zero=all_zero, plans=sass), indent=2) + '\n')
    if not all_zero:
        raise ValueError('a serving binary has FP64 SASS instructions')
    policy_path.write_text(json.dumps(policy, indent=2) + '\n')
    shutil.copy2(run / 'report.json', evidence / 'report.json')
    shutil.copy2(run / 'report.md', evidence / 'report.md')
    shutil.copy2(run / 'commands.sh', evidence / 'commands.sh')
    print(f'archived {len(policy["rounds"])} arms and {len(sass)} SASS audits')


if __name__ == '__main__':
    main()
