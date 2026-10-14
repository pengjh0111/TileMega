"""Audit accepted scheduler attempts without mixing rejected measurements."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .arms import sha256
from .identity_join import checked_join
from .report import discernible, metrics, summarize
from tilemega.build.identity import digest


def execution_ids(arm, command):
    result = {}
    for phase, key in (('decode', 'execution_identity'), ('prefill', 'prefill_execution_identity')):
        execution = arm.get(key)
        registered = command['binaries'].get(phase, {}).get('identity')
        if execution is None:
            if registered:
                raise ValueError('registered artifact has no execution identity: ' + phase)
            result[phase] = None  # Old reference binaries have no optional ABI.
            continue
        checked_join({'execution_identity': execution}, {'execution_identity': execution})
        if registered and execution['artifact_id'] != registered['artifact_id']:
            raise ValueError('executed artifact differs from registered binary: ' + phase)
        result[phase] = execution['execution_id']
    return result


def accepted_rounds(state_path, prefix, rounds=3):
    state = json.loads(Path(state_path).read_text())
    selected = {name: row for name, row in state.items() if name.startswith(prefix)}
    if not selected or any(row['status'] != 'done' for row in selected.values()):
        raise ValueError('comparison steps have not all completed successfully')
    groups, receipts, protocols = defaultdict(list), [], {}
    for name, row in sorted(selected.items()):
        folder = Path(row['out'])
        guard_path, round_path = folder / 'guard.jsonl', folder / 'round.json'
        guard = [json.loads(line) for line in guard_path.read_text().splitlines()]
        if not guard or guard[-1].get('phase') != 'after' or guard[-1].get('exit_code') != 0:
            raise ValueError('attempt lacks successful guard completion: ' + name)
        data = json.loads(round_path.read_text())
        if data.get('invalidated') or any(arm['exit_code'] for arm in data['arms'].values()):
            raise ValueError('attempt contains an invalid measurement: ' + name)
        protocol = dict(workload=data['workload'], prompt_sha256=data.get('prompt_sha256'),
                        step_bytes=data.get('step_bytes'), dram_gbps=data.get('dram_gbps'),
                        arms=sorted(data['arms']))
        if protocols.setdefault(data['cell'], protocol) != protocol:
            raise ValueError('workload, arms, prompts or bandwidth changed between rounds: ' + data['cell'])
        receipts.append(dict(step=name, attempt=row['attempts'], round=str(round_path),
            round_sha256=sha256(round_path), guard=str(guard_path), guard_sha256=sha256(guard_path)))
        for label, arm in data['arms'].items():
            command_path = folder / label / 'command.json'
            command = json.loads(command_path.read_text())
            binaries = {phase: item['sha256'] for phase, item in command['binaries'].items()}
            kind = command.get('kind', 'tm' if binaries else 'vllm')
            token_root = folder / label
            if kind == 'vllm':
                token_root /= 'B' + str(data['workload']['batch'])
                if not arm.get('vllm_version'):
                    raise ValueError('vLLM measurement lacks a version')
            elif kind != 'tm':
                raise ValueError('unsupported measured arm: ' + kind)
            tokens = token_root / f'tokens_N{data["workload"]["osl"]}_run1.json'
            groups[data['cell'], label].append(dict(round=data['round'],
                **metrics(arm, data['workload'], step_bytes=data.get('step_bytes'),
                          dram_gbps=data.get('dram_gbps')), binaries=binaries,
                execution_ids=execution_ids(arm, command), kind=kind,
                system_version=arm.get('vllm_version'),
                tokens_sha256=sha256(tokens), tokens_digest=digest(json.loads(tokens.read_text())),
                command_sha256=sha256(command_path)))
    cells = defaultdict(dict)
    for (cell, label), samples in sorted(groups.items()):
        if len(samples) != rounds or len({item['round'] for item in samples}) != rounds:
            raise ValueError('missing or duplicated comparison rounds: ' + cell + '/' + label)
        if any(item['binaries'] != samples[0]['binaries'] for item in samples):
            raise ValueError('binary identity changed between rounds: ' + cell + '/' + label)
        for field in ('execution_ids', 'kind', 'system_version'):
            if any(item[field] != samples[0][field] for item in samples):
                raise ValueError(field + ' changed between rounds: ' + cell + '/' + label)
        cells[cell][label] = dict(samples=samples,
            tpot_ms=summarize([item['tpot_seconds'] * 1000 for item in samples]),
            request_seconds=summarize([item['request_seconds'] for item in samples]),
            output_tokens_per_second=summarize([item['output_tokens_per_second'] for item in samples]),
            mbu=(summarize([item['mbu'] for item in samples]) if samples[0]['mbu'] is not None else None))
    for cell, arms in cells.items():
        rounds_seen = {label: {item['round'] for item in arm['samples']} for label, arm in arms.items()}
        if any(value != next(iter(rounds_seen.values())) for value in rounds_seen.values()):
            raise ValueError('unpaired comparison rounds: ' + cell)
        baselines = [label for label, arm in arms.items() if arm['samples'][0]['kind'] == 'vllm']
        for label, arm in arms.items():
            arm['throughput_ratios'] = {}
            for baseline in baselines:
                reference = {s['round']: s['output_tokens_per_second'] for s in arms[baseline]['samples']}
                arm['throughput_ratios'][baseline] = summarize([
                    s['output_tokens_per_second'] / reference[s['round']] for s in arm['samples']])
    return dict(counts=dict(Counter(row['status'] for row in state.values())),
                state_sha256=sha256(state_path), accepted_attempts=receipts, cells=dict(cells))


def compare(result, reference, candidate):
    rows = []
    for cell, arms in sorted(result['cells'].items()):
        a, b = arms[reference]['tpot_ms'], arms[candidate]['tpot_ms']
        rows.append(dict(cell=cell, reference=reference, candidate=candidate,
            reference_ms=a['median'], candidate_ms=b['median'],
            relative=b['median'] / a['median'] - 1,
            discernible=discernible(b['samples'], a['samples']),
            tokens_identical=len({sample['tokens_digest'] for label in (reference, candidate)
                for sample in arms[label]['samples']}) == 1))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--compare', nargs=2, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = accepted_rounds(args.state, args.prefix)
    result['comparisons'] = [row for pair in args.compare for row in compare(result, *pair)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(counts=result['counts'], comparisons=result['comparisons'])))


if __name__ == '__main__':
    main()
