#!/usr/bin/env python3
"""Wait for the B4 AB4 supplement, re-audit it and publish complete tables."""
import argparse
import datetime
from pathlib import Path
import subprocess
import time

from sm90d1_common import ROOT, read, sha, write as write_json
from sm90r6_merge_ab4_supplement import comparison_table, merge
from sm90r6_merge_analysis import write as write_csv


def publish(root, out, push=False):
    rows, processes, correctness, audits = merge(root)
    counts = dict(cells=len(audits), arms=len(rows), timing_processes=len(processes),
        psel_processes=sum(v['psel']['processes'] for v in audits),
        c2_processes=sum(v['c2_processes'] for v in correctness),
        c1_processes=sum(v['c1_processes'] for v in correctness))
    if counts != dict(cells=4, arms=8, timing_processes=24, psel_processes=32,
                      c2_processes=40, c1_processes=8):
        raise ValueError('incomplete supplement acceptance counts')
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        'SM90R6_ab4_b4.csv': rows,
        'SM90R6_ab4_b4_processes.csv': processes,
        'SM90R6_ab4_b4_correctness.csv': correctness,
        'SM90R6_ab4_comparison.csv': comparison_table(rows, ROOT / 'docs/results/SM90R6_ablation.csv'),
    }
    for name, values in tables.items():
        write_csv(out / name, values)
    report = (root / 'report-template.md').read_text().replace(
        'Results and final acceptance counts are inserted only after all cells pass the audits.', '')
    report += '\nVerified results (' + datetime.datetime.now(datetime.timezone.utc).isoformat() + '):\n\n'
    report += '| Model | AB4b TPOT (ms) | AB4c TPOT (ms) | AB4c change vs AB4b | Threshold | Decision |\n'
    report += '|---|---:|---:|---:|---:|---|\n'
    for model in sorted({v['model'] for v in rows}):
        pair = {v['arm']: v for v in rows if v['model'] == model}
        b, c = pair['AB4b'], pair['AB4c']
        report += (f'| {model} | {b["tpot_ms"]:.9f} | {c["tpot_ms"]:.9f} | '
                   f'{c["change_vs_ab4b_pct"]:+.6f}% | {c["threshold_pct"]:.6f}% | {c["decision"]} |\n')
    report += ('\nVerified acceptance: 32 P-SEL processes, 40/40 accepted C2 processes, '
               '8/8 C1 checks (256 positions per sequence, 1024 total per B4 check), '
               '24 final timing processes, and complete cross-arm timing-token equality. '
               'Rejected preflight/interference attempts are excluded and retained in raw logs.\n\n'
               'CSV values retain full floating-point precision. SM90R6_ab4_b4.csv includes all '
               'eight arm summaries, coordinates, predictions, build times, identities and provenance. '
               'SM90R6_ab4_b4_processes.csv retains the 24 accepted timing processes; '
               'SM90R6_ab4_b4_correctness.csv retains all eight correctness gates. '
               'SM90R6_ab4_comparison.csv combines the original B1/B16 and new B4 comparisons '
               'with explicit timing-protocol labels. Original B1/B16 data is unchanged.\n\n'
               'Reproduce in a new raw directory using scripts/sm90r6_prepare_ab4_b4.py, '
               'scripts/sm90r6_ab4_supplement.py, then scripts/sm90r6_merge_ab4_supplement.py. '
               'Use the retained search and compiler recorded in raw source.json. '
               'Raw environment.json records the H800 UUID/PCI mapping, CPU/system, '
               'CUDA/compiler and Python stack for this collection.\n')
    equivalence = root / 'olmoe-seed-equivalence.json'
    if equivalence.exists():
        pair = read(equivalence)
        a, b = (read(pair[k]) for k in ('initial_receipt', 'effective_receipt'))
        if [v['coordinates'] for v in a['finalists']] != [v['coordinates'] for v in b['finalists']]:
            raise ValueError('OLMoE seed equivalence receipt changed')
        report += ('\nVerified provenance correction: the initial OLMoE REF-R4-seed attempt '
                   'was superseded by the R5 A1 B4 seed, matching the original R6 seed family. '
                   'Their four finalist coordinates are equal. Final selection and correctness '
                   'use the corrected attempt; both raw attempts are retained.\n')
    report_path = out / 'SM90R6_ab4_b4.md'
    report_path.write_text(report)
    paths = [out / name for name in tables] + [report_path]
    write_json(root / 'publication.json', dict(verified=True, **counts,
        files={str(p): sha(p) for p in paths}, audit=str(root / 'summary/audit.json')))
    if push:
        if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'sm90r6':
            raise ValueError('publication requires the sm90r6 branch')
        relative = [str(p.relative_to(ROOT)) for p in paths]
        subprocess.run(['git', 'add', '--', *relative], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '--only', '-m', 'docs: publish audited B4 AB4 comparison data', '--', *relative],
                       cwd=ROOT, check=True)
        subprocess.run(['git', 'push', 'origin', 'sm90r6'], cwd=ROOT, check=True)
        write_json(root / 'publication-git.json', dict(commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), pushed=True))
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/results')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait:
        while not (root / 'summary/audit.json').exists():
            time.sleep(30)
    print(publish(root, args.out.resolve(), args.push), flush=True)
