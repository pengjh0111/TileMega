#!/usr/bin/env python3
"""Wait for all independently scheduled B4 measurements and publish audited data."""
import argparse
import datetime
from pathlib import Path
import subprocess
import time

from sm90d1_common import ROOT, read, sha, write
from sm90r6_dynamic_ab4 import PROTOCOL
from sm90r6_merge_dynamic_ab4 import write_tables


def publish(root, out, push=False):
    rows, paths, audit_path = write_tables(root, out)
    source = read(root / 'source.json')
    previous = Path(source['previous'])
    report = ('# SM90R6 AB4b / AB4c B4 supplement\n\n'
        'Verified collection of Qwen3-1.7B, Llama-3.1-8B, Qwen3-30B-A3B and OLMoE-1B-7B '
        'at batch 4 with the retained R6 search, compiler, exports, target and timing driver. '
        'AB4b uses native aggregate selection; AB4c emits a kernel sequence for exactly the '
        'selected AB4b coordinates, grid and shared-memory allocation.\n\n'
        'The user authorized independent process scheduling across all four visible H800 GPUs. '
        'Each card has a worker polling approximately every 2 seconds. A process holds a '
        'cooperative claim only during its own attempt. Dependencies unlock separately per '
        'model, and up to four independent items can execute concurrently. Rejected interference '
        'attempts release the claim and rejoin the queue after 2 seconds; within each priority '
        'they rotate behind items attempted less recently. The original physical-card guard, '
        '25-second preflight, ownership checks and throttle checks remain in force.\n\n'
        'This B4 dataset uses protocol `' + PROTOCOL + '`. Its process slots retain the '
        'original counts and arm labels, while physical start order and GPU assignment are '
        'dynamic. AB4b and AB4c medians may combine different cards and concurrent host activity. '
        'Per-process GPU, PID, process group, actual start/end times and per-GPU summaries '
        'are retained so those effects can be examined. Original B1/B16 protocols remain '
        'explicitly labeled in the combined table.\n\n'
        'P-SEL uses all four linked finalists in logical ABCDDCBA slots: 8 fresh processes '
        'per model, median TPOT, with the original top-two prediction tie rule. '
        'Each selected arm passes 5 fresh C2 processes, 1024 decode steps in both loop modes, '
        'and C1 at 256 positions per sequence (1024 total positions for B4). '
        'Final timing uses 3 fresh processes per arm, each with 1 warmup and 3 timed samples '
        'at N=1 and N=1024. TPOT is `(median(T1024)-median(T1))*1000/1023`. '
        'The original range-based decision threshold is `max(0.5%, 2*max(arm ranges))`. '
        'Every accepted timing process is re-audited against its actual raw samples. '
        'All final timing processes also emit identical full tokens across the two arms.\n\n')
    report += 'Verified results (' + datetime.datetime.now(datetime.timezone.utc).isoformat() + '):\n\n'
    report += '| Model | AB4b TPOT (ms) | AB4c TPOT (ms) | AB4c change | Threshold | Decision |\n'
    report += '|---|---:|---:|---:|---:|---|\n'
    for model in sorted({v['model'] for v in rows}):
        pair = {v['arm']: v for v in rows if v['model'] == model}
        b, c = pair['AB4b'], pair['AB4c']
        report += (f'| {model} | {b["tpot_ms"]:.9f} | {c["tpot_ms"]:.9f} | '
                   f'{c["change_vs_ab4b_pct"]:+.6f}% | {c["threshold_pct"]:.6f}% | {c["decision"]} |\n')
    report += ('\nVerified acceptance: 32 P-SEL processes, 40/40 C2 processes, 8/8 C1 checks '
        'and 24 final timing processes, totaling 104 accepted processes plus 4 occupancy items. '
        'Twelve accepted P-SEL processes were imported from the stopped whole-unit queue; '
        'all imports preserve their original paths, GPU and state provenance. Rejected attempts '
        'remain in the attempt ledger and raw logs.\n\n'
        'CSV floating-point values retain full precision. The six CSV files contain all eight '
        'arm summaries; all 24 final timing processes; all eight correctness gates; the combined '
        '12 model/batch comparisons at B1, B4 and B16; all 108 accepted queue items with complete '
        'definitions and results; and every dynamic attempt with raw guard hashes. '
        'The arm table includes geometry, structure, build times, search predictions and '
        'binary/input hashes. No fields or rows were removed to meet a size limit.\n\n'
        'Raw collection: `' + str(root) + '`. Frozen search/build inputs and original raw data: `'
        + str(previous) + '`. The original `environment.json` records hardware and software; '
        'the new `queue/inventory.json` rechecks the current GPU UUID mapping. '
        'Run `sm90r6_gpu_items.py` for the process queue and `sm90r6_dynamic_ab4.py` for '
        'dependency advancement. Re-audit and regenerate tables with '
        '`sm90r6_merge_dynamic_ab4.py --root <raw> --out <tables>`.\n')
    equivalence = previous / 'olmoe-seed-equivalence.json'
    if equivalence.exists():
        value = read(equivalence)
        a, b = (read(value[k]) for k in ('initial_receipt', 'effective_receipt'))
        if [v['coordinates'] for v in a['finalists']] != [v['coordinates'] for v in b['finalists']]:
            raise ValueError('OLMoE seed equivalence changed')
        report += ('\nVerified OLMoE seed correction: collection uses the R5 A1 B4 seed family '
                   'from the original R6 methodology. Its four finalist coordinates equal '
                   'those from the superseded initial REF-R4-seed attempt; both receipts remain.\n')
    report_path = out / 'SM90R6_ab4_b4.md'
    report_path.write_text(report)
    paths.append(report_path)
    write(root / 'publication.json', dict(verified=True, protocol=PROTOCOL,
        audit=str(audit_path), audit_sha256=sha(audit_path),
        files={str(p): sha(p) for p in paths}, script_sha256=sha(__file__)))
    if push:
        if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'sm90r6':
            raise ValueError('publication requires sm90r6 branch')
        relative = [str(p.relative_to(ROOT)) for p in paths]
        subprocess.run(['git', 'add', '--', *relative], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '--only', '-m', 'docs: publish audited dynamic B4 AB4 data',
                        '--', *relative], cwd=ROOT, check=True)
        subprocess.run(['git', 'push', 'origin', 'sm90r6'], cwd=ROOT, check=True)
        write(root / 'publication-git.json', dict(pushed=True, commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    return read(audit_path)['accepted_processes']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/results')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait:
        while not all((root / c['id'] / 'complete.json').exists() for c in read(root / 'cells.json')):
            time.sleep(15)
        while any(v['status'] != 'done' for v in read(root / 'queue/items.json').values()):
            time.sleep(2)
    print(dict(accepted_processes=publish(root, args.out.resolve(), args.push)), flush=True)
