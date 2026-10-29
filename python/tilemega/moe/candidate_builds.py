"""Prepare the original four MoE candidates and their own native prompt pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .cost_provenance import Files, need, validate_complete_scope


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(preview_path, domain_tool, target, out):
    preview_path, target, out = map(lambda p: Path(p).resolve(), (preview_path, target, out))
    need(not out.exists(), 'existing candidate preparation is preserved')
    files = Files()
    preview_ref, target_ref = record(preview_path), record(target)
    files.check(preview_ref)
    files.check(target_ref)
    preview = json.loads(preview_path.read_text())
    need(preview.get('kind') == 'mx5_moe_candidate_preview' and preview.get('schema_version') == 1,
         'native MoE candidate preview required')
    need(preview.get('layers') == 24 and preview.get('stages_per_step') == 145,
         'candidate preview is not MoE-L24')
    need(all(preview.get(name) is False for name in
             ('execution_authorized', 'file_qualification_implied', 'solver_selected', 'final', 'default_enabled')),
         'preview grants unverified model or execution policy')
    producer, bridge, table = (files.check(preview[name]) for name in ('producer', 'input', 'stage_table'))
    request = preview['request']
    need(request == dict(batch=preview['batch'], threads=preview['threads'], grid=104,
                        capacity=1088, past_lo=64, past_hi=1086), 'preview request differs')
    evidence = validate_complete_scope(table, domain_tool)
    need((evidence['batch'], evidence['threads']) == (preview['batch'], preview['threads'])
         and evidence['table'] == preview['stage_table'], 'file-qualified scope differs from preview')
    candidates = preview['candidates']
    need([c['index'] for c in candidates] == [0, 1, 2, 3]
         and all(c['prompt_pair_required'] is True for c in candidates), 'four original prompt pairs required')
    base = [str(producer), 'compile', str(bridge)]
    workload = ['--emit', 'serving', '--frontend', 'decoder', '--batch', str(preview['batch']),
        '--prompt-len', '64', '--capacity', '1088', '--past-range', '64:1086',
        '--runtime-target', str(target)]
    # Re-run the actual native ranking: edited ranks, costs or pins cannot be
    # turned into executable candidates by this host preparation helper.
    with tempfile.TemporaryDirectory(prefix='mx5_moe_preview_') as temporary:
        output = Path(temporary) / 'plan.so'
        argv = base + [str(output), '--serving', 'decode'] + workload + [
            '--mx-decode-v2-solve', str(table), '--mx-decode-v2-threads', str(preview['threads']),
            '--selection', 'predicted']
        process = subprocess.run(argv, capture_output=True, text=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES=''))
        need(process.returncode == 0, 'original native preview regeneration failed: ' + process.stderr)
        need(not output.exists() and json.loads(Path(str(output) + '.mx5_candidates.json').read_text()) == preview,
             'preview differs from original native top-four ranking')
    files.unchanged()
    out.mkdir(parents=True)
    jobs = []
    for candidate in candidates:
        index = candidate['index']
        pin = out / ('candidate%d.pin.json' % index)
        write(pin, candidate['pin'])
        for watchdog in (0, 1):
            for phase in ('prefill', 'decode'):
                binary = out / ('candidate%d_w%d' % (index, watchdog)) / phase / 'plan.so'
                argv = base + [str(binary), '--serving', phase] + workload + [
                    '--evaluate-configs', str(pin), '--selection', 'predicted', '--watchdog', str(watchdog)]
                if phase == 'prefill':
                    argv += ['--mx-prompt', '1']
                jobs.append(dict(index=index, phase=phase, watchdog=watchdog, binary=str(binary),
                    pin=record(pin), argv=argv, native_cpu_build_only=True))
    result = dict(kind='mx5_moe_original_candidate_builds', schema_version=1,
        preview=preview_ref, runtime_target=target_ref, producer=preview['producer'],
        bridge=preview['input'], stage_table=preview['stage_table'], file_evidence=evidence,
        batch=preview['batch'], threads=preview['threads'], jobs=jobs,
        matching_prompt_pairs=8, GPU_calls=0, solver_selected=False,
        grants_512_exception=False, final=False, default_enabled=False)
    write(out / 'build_jobs.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', required=True, type=Path)
    parser.add_argument('--domain-tool', required=True, type=Path)
    parser.add_argument('--runtime-target', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.preview, args.domain_tool, args.runtime_target, args.out)


if __name__ == '__main__':
    main()
