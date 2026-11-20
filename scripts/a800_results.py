#!/usr/bin/env python3
"""Collect guarded requests from one A800 hardware class, with per-round UUIDs."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics

from a100_baselines import BATCHES, ENGINES, MODELS, ROOT, inspect_inputs, sha
from a100_results import measurement, read, write_csv
from a800_baselines import INVENTORY, RUN, hardware_class
from a800_mpk_preflight import REVISION as MPK_REVISION


REPORT_ENGINES = (*ENGINES, 'mpk')
METRICS = ('tpot_ms', 't1_s', 't1024_s', 'tok_s')
OOM_MARKERS = ('CUDA out of memory', 'torch.OutOfMemoryError', 'CUDA_ERROR_OUT_OF_MEMORY')


def metrics(rows, count):
    if len(rows) != count:
        return dict.fromkeys(METRICS, '')
    return {name: statistics.median(row[name] for row in rows) for name in METRICS}


def summarize(rows, key):
    values = sorted({row[key] for row in rows})
    if len(values) <= 1:
        return values[0] if values else ''
    return json.dumps([json.loads(value) for value in values])


def baseline_scope(cohorts, inputs):
    scope = {}
    for run in cohorts:
        recorded = read(run/'baseline_inputs.json')
        if not recorded:
            continue
        if recorded['models'] != inputs['models']:
            raise ValueError(f'baseline and native inputs differ for {run.name}')
        if hardware_class(recorded['device']) != hardware_class(inputs['device']):
            continue
        devices = {row['guard_policy_sha256']: row['device']
                   for row in (read(path) for path in run.glob('allocations/*/baseline_inputs.json'))}
        state = read(run/'baseline/state.json')
        capacity_files = {path for path in run.glob('*.oom.json') if path.is_file()}
        for job in read(run/'queue/queue_baselines.json'):
            if 'model' not in job:
                continue
            name = job['name']
            step = state.get(name, {})
            capacity = run/f'{name}.oom.json'
            if step.get('out') or capacity in capacity_files:
                key = job['model'], job['engine'], job['batch']
                identity = read(Path(step['out'])/'job/physical-device-identity.json') if step.get('out') else {}
                device = devices.get(identity.get('guard_policy_sha256'))
                if step.get('status') == 'done' and (not device or
                        hardware_class(device) != hardware_class(inputs['device'])):
                    raise ValueError(f'no matching allocation receipt for {name}')
                scope.setdefault(key, []).append((run, job, step, capacity, device or recorded['device']))
    return scope


def confirmed_oom(run, job, step, capacity, inputs):
    receipt = read(capacity)
    if receipt:
        return (receipt['status'] == 'OOM' and
                receipt['device_bytes'] == inputs['device']['memory_mib'] * 2**20)
    if step.get('status') != 'failed':
        return False
    log = run/'baseline'/f'{job["name"]}.log'
    latest = log.read_text(errors='replace').rsplit('\nATTEMPT ', 1)[-1] if log.is_file() else ''
    return any(mark in latest for mark in OOM_MARKERS)


def trtllm_olmoe_rejection(jobs):
    for run, job, step, _, _ in jobs:
        if step.get('status') != 'failed':
            continue
        log = run/'baseline'/f'{job["name"]}.log'
        latest = log.read_text(errors='replace').rsplit('\nATTEMPT ', 1)[-1] if log.is_file() else ''
        if ('TensorRT-LLM version: 0.21.0' in latest and
                'Unknown architecture for AutoModelForCausalLM: OlmoeForCausalLM' in latest):
            return str(log.relative_to(RUN))
    return ''


def baseline_cell(inputs, jobs, model, engine, batch, unsupported=''):
    accepted, oom_rounds, evidence, harnesses = {}, set(), set(), set()
    for run, job, step, capacity, device in jobs:
        repeat = job['round']
        harnesses.add(job['harness_sha256'])
        if step.get('out'):
            evidence.add(str(Path(step['out']).relative_to(RUN)))
        if row := measurement(job, step, dict(inputs, device=device)):
            if engine == 'mpk':
                proof = read(run/'mpk-accepted.json')
                if proof.get('measurement_sha256') != row['receipt']:
                    continue
                row['reference_rows'] = proof.get('matching_first_tokens', 0)
            row['gpu_uuid'] = device['uuid']
            row['harness_sha256'] = job['harness_sha256']
            if repeat in accepted and accepted[repeat] != row:
                raise ValueError(f'duplicate accepted round for {job["name"]}; select one cohort')
            accepted[repeat] = row
        elif confirmed_oom(run, job, step, capacity, inputs):
            oom_rounds.add(repeat)
            if capacity.is_file():
                evidence.add(str(capacity.relative_to(RUN)))
    rows = [accepted[key] for key in sorted(accepted)]
    needed = 3
    status = ('measured' if len(rows) == needed else 'incomplete' if rows else
              'N/A' if unsupported else 'OOM' if len(oom_rounds) == needed else
              'attempted' if evidence else 'pending')
    if status == 'N/A':
        evidence.add(unsupported)
    correctness = 'exact token counts' if rows else 'pending'
    if engine == 'mpk' and rows:
        checked = min(row['reference_rows'] for row in rows)
        correctness = f'stable greedy repeats, independent first-token agreement on {checked}/{batch} rows'
    return rows, dict(model=model, framework=engine, batch=batch, status=status, processes=len(rows),
        required_processes=needed, correctness=correctness,
        **metrics(rows, needed), backend=summarize(rows, 'backend'), versions=summarize(rows, 'versions'),
        gpu_uuid=' '.join(sorted({row['gpu_uuid'] for row in rows})),
        config_sha256=inputs['models'][model]['config_sha256'],
        prompt_sha256=inputs['models'][model]['prompt_sha256'], checkpoint_sha256='',
        harness_sha256=' '.join(sorted({row['harness_sha256'] for row in rows} or harnesses)),
        receipt_sha256=' '.join(row['receipt'] for row in rows), evidence=' '.join(sorted(evidence)))


def native_cell(inputs, model, batch, selection=None):
    def compatible(receipt):
        if 'device' in receipt:
            return hardware_class(receipt['device']) == hardware_class(inputs['device'])
        return receipt['device_uuid'] == inputs['device']['uuid']

    root = RUN/'native'
    cell = f'*/{model}/B{batch}'
    checks = [read(path) for pattern in ('correctness-*.json', '*/correctness-*.json')
              for path in root.glob(f'{cell}/{pattern}')]
    checks = [row for row in checks if compatible(row) and row.get('pass')]
    def pair(receipt):
        return receipt['prefill_sha256'], receipt['decode_sha256'], receipt['target_sha256']

    checks_by_pair = defaultdict(list)
    for check in checks:
        checks_by_pair[pair(check)].append(check)
    ready = {key for key, checks in checks_by_pair.items()
             if len({r['round'] for r in checks}) >= 5
             and len({r['tokens_sha256'] for r in checks}) == 1
             and any(r.get('hf_pass') for r in checks)}
    needed = 3 if model in tuple(MODELS)[:4] else 5
    if selection is None:
        receipts = [read(path) for path in sorted(root.glob(f'{cell}/*/measure-*.json'))]
        receipts = [row for row in receipts if compatible(row)]
    else:
        if selection not in ready:
            raise ValueError(f'{model} B{batch} selected artifacts lack exact-pair correctness')
        receipts = [read(path) for path in sorted(root.glob(f'{cell}/*/candidate-timing-*.json'))]
        receipts = [row for row in receipts if compatible(row) and pair(row) == selection]
        if {row['round'] for row in receipts} != set(range(needed)):
            raise ValueError(f'{model} B{batch} selected artifacts lack independent timing rounds')
    rows, artifact_ids = [], set()
    for receipt in receipts:
        if not receipt.get('pass'):
            raise ValueError('native measurement receipt did not pass')
        attempt = Path(receipt['evidence']).parent.parent
        job = dict(name=f'{model}-native-r{receipt["round"]}', model=model, batch=batch,
                   round=receipt['round'])
        device = receipt.get('device', inputs['device'])
        row = measurement(job, dict(status='done', out=str(attempt)), dict(inputs, device=device))
        rows.append(dict(row, gpu_uuid=device['uuid']))
        artifact_ids.add(pair(receipt))
    if len(artifact_ids) > 1:
        raise ValueError(f'{model} B{batch} contains more than one native plan pair')
    if len(rows) > needed or len({r['round'] for r in receipts}) != len(rows):
        raise ValueError('choose an explicit cohort without extra formal rounds')
    chosen = artifact_ids or ready
    artifacts = next(iter(chosen)) if len(chosen) == 1 else ('', '', '')
    accepted = checks_by_pair.get(artifacts, checks)
    status = ('measured' if len(rows) == needed else 'incomplete' if rows else
              'pending_timing' if ready else 'pending_correctness')
    uuids = {row['gpu_uuid'] for row in rows} or {row['device_uuid'] for row in accepted}
    return rows, dict(model=model, batch=batch, status=status, processes=len(rows), required_processes=needed,
        correctness='five fresh checks passed' if rows or ready else 'pilot passed' if checks else 'pending',
        accepted_correctness_processes=len(accepted),
        **metrics(rows, needed), backend='SM80 native' if rows or ready else '', fastest_baseline='',
        fastest_baseline_tpot_ms='', speedup='', config_sha256=inputs['models'][model]['config_sha256'],
        prompt_sha256=inputs['models'][model]['prompt_sha256'], checkpoint_sha256='',
        prefill_sha256=artifacts[0], decode_sha256=artifacts[1], target_sha256=artifacts[2],
        gpu_uuid=' '.join(sorted(uuids)), versions=summarize(rows, 'versions'),
        receipt_sha256=' '.join(row['receipt'] for row in rows))


def request_times(cell, rows, tpot_only=False):
    if not rows:
        return cell['status'] if cell['status'] in {'N/A', 'OOM'} else '—'
    values = metrics(rows, len(rows))
    times = f"{values['tpot_ms']:.3f}"
    if not tpot_only:
        times = f"{1000 * values['t1_s']:.1f} / {values['t1024_s']:.3f} / {times}"
    if cell['status'] == 'incomplete':
        times += f" ({len(rows)}/{cell['required_processes']})"
    return times


def write_time_table(path, cells, tpot_only=False):
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    if tpot_only:
        lines = ['# A800 单卡 decode TPOT', '', f'采集时间 {now}。单位为 ms/token。',
                 '输入 64 tokens，输出 1024 tokens，TP1，BF16。',
                 '每个进程取 (median(T1024) − median(T1)) / 1023，再取进程间中位数。']
    else:
        lines = ['# A800 单卡执行时间', '', f'采集时间 {now}。每格依次为 **T1（ms） / T1024（s） / TPOT（ms/token）**。',
                 'T1 和 T1024 是完整请求的墙钟时间，包含处理同一份 64-token 输入。每个长度先热身一次，再取三次请求的中位数。',
                 '每个独立进程用 (T1024 − T1) / 1023 计算 TPOT，三列分别取进程间中位数。']
    lines.extend(['括号给出部分结果的已完成进程数和计划进程数。— 表示尚无有效计时，N/A 表示已确认不支持，OOM 表示容量不足。',
                  '', '| 模型 | Batch | TileMega | vLLM | SGLang | TRT-LLM | torch.compile | MPK |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|'])
    names = {'qwen3_0_6b': 'Qwen3-0.6B', 'qwen3_1_7b': 'Qwen3-1.7B',
             'llama3_2_1b': 'Llama-3.2-1B', 'llama3_2_3b': 'Llama-3.2-3B',
             'llama3_1_8b': 'Llama-3.1-8B', 'qwen3_30b_a3b': 'Qwen3-30B-A3B',
             'mistral_7b_v0_3': 'Mistral-7B-v0.3', 'olmoe_1b_7b': 'OLMoE-1B-7B'}
    for model in MODELS:
        for batch in BATCHES:
            values = [request_times(*cells[model, batch, engine], tpot_only=tpot_only)
                      for engine in ('native', *REPORT_ENGINES)]
            lines.append('| ' + ' | '.join([names[model], str(batch), *values]) + ' |')
    temporary = path.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', action='append', required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--native-selection', type=Path,
                   help='JSON artifact triples keyed by model/B<batch> for promoted timing cohorts')
    p.add_argument('--out', type=Path, default=ROOT/'docs/results')
    args = p.parse_args()
    if any(not name or not all(c.isascii() and (c.isalnum() or c in '-_') for c in name)
           for name in args.cohort):
        p.error('invalid cohort name')
    selected = {}
    if args.native_selection:
        selections = json.loads(args.native_selection.read_text())
        if not isinstance(selections, dict):
            p.error('native selection must be a JSON object')
        valid_cells = {f'{model}/B{batch}' for model in MODELS for batch in BATCHES}
        for cell, artifact in selections.items():
            if cell not in valid_cells:
                p.error(f'unknown native cell {cell}')
            if not isinstance(artifact, dict) or not all(isinstance(artifact.get(key), str) for key in
                    ('prefill_sha256', 'decode_sha256', 'target_sha256')):
                p.error(f'incomplete native artifact identity for {cell}')
            selected[cell] = (artifact['prefill_sha256'], artifact['decode_sha256'], artifact['target_sha256'])
    dev = read(args.target)['device_provenance']
    if 'A800' not in dev['name']:
        p.error('supply the guarded A800 target')
    inputs = dict(device=dev, models=inspect_inputs(INVENTORY))
    scope = baseline_scope((RUN/'cohorts'/name for name in args.cohort), inputs)
    mpk_path = RUN/'mpk/preflight.json'
    mpk = read(mpk_path)
    unavailable = {}
    if mpk:
        if (mpk.get('revision') != MPK_REVISION or
                mpk.get('target_sha256') != sha(args.target)):
            raise ValueError('MPK capability receipt has a different source or target')
        for entry in mpk['cells']:
            model, batch = entry['model'], entry['batch']
            if entry['config_sha256'] != inputs['models'][model]['config_sha256']:
                raise ValueError(f'MPK capability receipt has a different config for {model}')
            if entry['status'] == 'N/A' and entry['reason']:
                unavailable[model, batch] = str(mpk_path.relative_to(RUN))
    trt_olmoe = trtllm_olmoe_rejection(scope.get(('olmoe_1b_7b', 'trtllm', 1), []))
    baseline, native = [], []
    time_cells = {}
    for model in MODELS:
        all_rows, checkpoint_hashes = [], set()
        for batch in BATCHES:
            measured = []
            for engine in REPORT_ENGINES:
                unsupported = trt_olmoe if (model, engine) == ('olmoe_1b_7b', 'trtllm') else ''
                if engine == 'mpk':
                    unsupported = unavailable.get((model, batch), '')
                rows, cell = baseline_cell(inputs, scope.get((model, engine, batch), []),
                                           model, engine, batch, unsupported)
                checkpoint_hashes.update(row['checkpoint_sha256'] for row in rows)
                time_cells[model, batch, engine] = (cell, rows)
                measured.append(cell)
                baseline.append(cell)
                all_rows.append(cell)
            rows, cell = native_cell(inputs, model, batch, selected.get(f'{model}/B{batch}'))
            checkpoint_hashes.update(row['checkpoint_sha256'] for row in rows)
            if all(row['status'] in {'measured', 'OOM', 'N/A'} for row in measured):
                valid = [r for r in measured if r['status'] == 'measured']
                if valid:
                    fastest = min(valid, key=lambda row: row['tpot_ms'])
                    cell.update(fastest_baseline=fastest['framework'],
                                fastest_baseline_tpot_ms=fastest['tpot_ms'])
                    if cell['status'] == 'measured':
                        cell['speedup'] = fastest['tpot_ms']/cell['tpot_ms']
            time_cells[model, batch, 'native'] = (cell, rows)
            native.append(cell)
            all_rows.append(cell)
        if len(checkpoint_hashes) > 1:
            raise ValueError(f'shared checkpoint changed for {model}')
        for row in all_rows:
            row['checkpoint_sha256'] = next(iter(checkpoint_hashes), '')
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'A800_baselines.csv', baseline)
    write_csv(args.out/'A800_llm.csv', native)
    write_time_table(args.out/'A800_execution_times.md', time_cells)
    write_time_table(args.out/'A800_TPOT.md', time_cells, tpot_only=True)


if __name__ == '__main__':
    main()
