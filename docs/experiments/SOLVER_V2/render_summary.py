#!/usr/bin/env python3
"""Rebuild the sixteen R9 report sections; incomplete evidence stays explicit.

Runs report_tables.py and every verify.py gate first. A rendered report is
never a substitute for a passing gate, a complete matrix, or final review.
"""
import argparse
import csv
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import time

E = pathlib.Path(__file__).resolve().parent
R = E.parents[2]


def table(name):
    with (E / 'report_tables' / (name + '.tsv')).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def md_table(records, fields):
    if not records:
        return '**Pending: no complete evidence rows.**\n'
    def cell(value):
        if isinstance(value, float):
            value = f'{value:.9g}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    lines += ['| ' + ' | '.join(cell(r.get(f, 'pending')) for f in fields) + ' |' for r in records]
    return '\n'.join(lines) + '\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-matrix',action='store_true')
    args=parser.parse_args()
    if args.wait_for_matrix:
        while True:
            state_file=E/'matrix_progress.json'
            output=E/'verify_after_matrix.log'
            state=json.loads(state_file.read_text()) if state_file.exists() else None
            if state and not state['pending'] and not state['reference_pending'] and output.exists():
                raw=output.read_text()
                if re.search(r'^VERIFY hard_failed=.*$',raw,re.M) and output.stat().st_mtime_ns>=state['observed_ns']:
                    break
            time.sleep(15)
    subprocess.run(['python3', str(E / 'report_tables.py')], cwd=R, check=True)
    with (E / 'verify_report.log').open('w') as log:
        result = subprocess.run(['python3', str(E / 'verify.py')], cwd=R,
                                stdout=log, stderr=subprocess.STDOUT)
    verify = (E / 'verify_report.log').read_text()
    verdicts = {m[1]: (m[2], m[3]) for m in re.finditer(r'^(G-\d+) (PASS|FAIL) (.*)$', verify, re.M)}
    if len(verdicts) != 13 or len(re.findall(r'^C-\d+ (?:PASS|FAIL)', verify, re.M)) != 20:
        raise RuntimeError('verifier did not finish every gate; refusing to render a complete report')
    baseline = json.loads((E / 'baseline.json').read_text())
    prompt = pathlib.Path(baseline['prompt_path'])
    prompt_hash = hashlib.sha256(prompt.read_bytes()).hexdigest()
    if prompt_hash != baseline['prompt_sha256']:
        raise RuntimeError('prompt changed from the recorded baseline')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip()
    commits = subprocess.check_output(['git', 'log', '--reverse', '--format=%h %s',
                                      baseline['baseline'] + '..HEAD'], cwd=R, text=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', baseline['baseline'], '--'], cwd=R, text=True).splitlines()
    perf, phase, status = table('performance'), table('phases'), table('status')
    resources, classes = table('resources'), table('classes')
    complete = len(perf) == 40
    state_path=E/'matrix_progress.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    terminal=bool(state) and not state.get('pending') and not state.get('reference_pending')
    pieces = ['# TileMega R9 — solver reconstruction\n',
              ('**Measurement matrix complete; gate results below remain authoritative. Final review and push are separate.**\n'
               if complete else f'**Matrix terminal with {len(perf)}/40 valid measured arms; failed items require review, not a PASS declaration.**\n'
               if terminal else f'**IN PROGRESS — {len(perf)}/40 measured arms. This is not the final R9 delivery.**\n')]
    def section(number, title, text):
        pieces.append(f'## {number}. {title}\n\n{text}\n')
    section(1, 'Provenance and commits',
            f'Generated UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}.\n\n'
            f'Baseline: `{baseline["baseline"]}`. Report source HEAD: `{head}`.\n\n'
            f'Prompt: `{prompt}`; SHA256 `{prompt_hash}`. TODO update was already applied '
            f'in baseline commit `{baseline["todo_update_commit"]}`.\n\n````text\n{commits}````')
    section(2, 'Complete verifier output',
            f'Fresh raw-evidence verification exited {result.returncode}. '
            'The verifier reads source, logs and dumps, not this report or report tables. '
            'Missing evidence produces FAIL; a pending experiment is not a measured negative result.\n\n'
            f'````text\n{verify}````')
    section(3, 'Gates', md_table([dict(gate=f'G-{i}', result=verdicts[f'G-{i}'][0],
            evidence=f'verify_report.log: G-{i}', type=('research' if i == 8 else 'hard' if i <= 7 else 'report'))
            for i in range(1, 14)], ['gate', 'type', 'result', 'evidence']))
    stops = []
    for path in sorted((E / 'legacy_r8_domain').glob('*/downstream_blocked.json')):
        stops.append(dict(cell=path.parent.name, reason=json.loads(path.read_text()).get('reason'),
                          evidence=str(path.relative_to(E))))
    for directory in sorted((E/'matrix').glob('*/*')):
        if not directory.is_dir():continue
        for name in ('solve.command.json','measure.command.json','oracle_audit.command.json'):
            path=directory/name
            if path.exists() and json.loads(path.read_text()).get('exit_code',0):
                stops.append(dict(cell=str(directory.relative_to(E)),reason=f'{name}: nonzero exit; inspect raw log before diagnosing',evidence=str(path.relative_to(E))))
    section(4, 'Stopped items and degraded forms',
            (md_table(stops, ['cell', 'reason', 'evidence']) if stops else 'No active control downstream-block marker at render time.\n') +
            '\nSearches still running or awaiting admission are unfinished work, not stopped items. '
            'The Qwen seq16 pre-output OOM attempt is archived separately; recovery reran all ten processes.\n\n' +
            md_table([r for r in status if r['state'] != 'measured'], ['model', 'seq', 'arm', 'state', 'evidence']) +
            '\nDeclared forms: GEMM-only coordinate search; diagnostic reference-domain regression; '
            'legacy R8 five-shape control; parallel child evaluation; warm physical resource cache. '
            'None substitutes for anchored coverage. Full details follow.')
    section(5, 'Deviation declarations', (E / 'deviations.md').read_text())
    totals = [p for p in phase if p['phase'] in ('total', 'import', 'variant_compile', 'megakernel_compile', 'oracle_general')]
    section(6, 'Solver latency and cache accounting',
            'The full phase decomposition is [phases.tsv](report_tables/phases.tsv). '
            'Parallel phase sums can exceed wall-clock total; admission queue time is separate. '
            'Legacy and skeleton search domains differ as declared above.\n\n' +
            md_table(totals, ['model', 'seq', 'arm', 'phase', 'count', 'total_ms']) + '\n' +
            md_table(table('derive_comparison'), ['model', 'split', 'mode', 'count', 'total_ms', 'cache_hit', 'cache_miss', 'bytes_equal']) +
            '\nFull-search cache statistics:\n\n' + md_table(table('searches'),
            ['model', 'seq', 'arm', 'evaluations', 'rounds', 'jobs', 'cache_hit', 'cache_miss', 'cache_hit_share']))
    indexed = {(p['model'], p['seq'], p['arm']): p for p in perf}
    comparisons = {(p['model'], p['seq'], p['arm']): p for p in table('comparisons')}
    full = []
    for s in status:
        key = s['model'], s['seq'], s['arm']
        row = dict(s, **{k: v for k, v in indexed.get(key, {}).items() if k not in s})
        row['l2_legacy'] = 1 if s['arm'] == 'legacy' and key in indexed else comparisons.get(key, {}).get('l2_legacy', 'pending')
        full.append(row)
    section(7, 'Eight cells by five arms', md_table(full,
            ['model', 'seq', 'arm', 'state', 'l05_ms', 'l1_ms', 'l2_ms', 'l2_l1', 'l2_l05', 'l2_legacy']))
    compact = {}
    for row in classes:
        compact.setdefault(tuple(row[f] for f in ('model', 'seq', 'arm', 'class')), row)
    section(8, 'Per-class geometry, uniform seed and residency', md_table(list(compact.values()),
            ['model', 'seq', 'arm', 'class', 'tile_m', 'tile_n', 'tile_k', 'stages', 'split_k',
             'seed_m', 'seed_n', 'seed_k', 'seed_stages', 'seed_split']) +
            '\nAll GEMM-to-class mappings: [classes.tsv](report_tables/classes.tsv).\n\n' +
            md_table(table('placements'), ['model', 'seq', 'arm', 'variants', 'residency', 'grid']))
    section(9, 'Top-five resource estimates and driver results', md_table(resources,
            ['model', 'seq', 'arm', 'rank', 'estimated', 'actual', 're_solved', 'residency']) +
            '\nCandidate keys and raw evidence paths: [resources.tsv](report_tables/resources.tsv).')
    section(10, 'Symbolic edge census and exact Oracle validation',
            'Semantic CG edges and physical executor-order edges are separate populations. '
            'G-5 audits the GPU-measured winner, including CG hash and actual grid/residency/κ.\n\n' +
            md_table(table('oracles'), ['model', 'seq', 'arm', 'graph', 'field', 'category', 'count']) +
            '\nSemSig collision and byte-equality evidence:\n\n````text\n' +
            (E / 'cache_test.log').read_text() + '````\n\n' +
            md_table([p for p in phase if p['phase'] == 'oracle_general'], ['model', 'seq', 'arm', 'count', 'total_ms']))
    section(11, 'Placement destinations and interleaving', md_table(table('placements'),
            ['model', 'seq', 'arm', 'affinity_share', 'home_share', 'spread_other_share',
             'average_candidates', 'interleaving', 'legacy_eft_interleaving']) +
            '\nEvery worker, including idle workers where grid is known: '
            '[worker_queues.tsv](report_tables/worker_queues.tsv). Aggregate ratios use total '
            'transitions / total adjacent slots, not the mean of worker ratios.')
    section(12, 'Loss from narrowing', md_table(table('comparisons'), ['model', 'seq', 'arm', 'l2_kW', 'l2_legacy']))
    section(13, 'Level-2 versus simulator ranking', md_table(resources,
            ['model', 'seq', 'arm', 'rank', 'level2_ns', 'simulated_ns', 'level2_rank', 'simulated_rank']) +
            '\nThese scores use each top-five candidate after real occupancy re-solving when needed. '
            'The outer-loop pre-recheck score is retained in each arm’s raw search.tsv.')
    section(14, 'Attention attribution at seq16/64',
            'The simulator field reconstructed here is dependency-only node-duration CP. '
            'Its attention fraction is not a measured arithmetic-utilization fraction.\n\n' +
            md_table([p for p in table('paths') if p['seq'] in ('16', '64')],
                     ['model', 'seq', 'arm', 'cp_ns', 'path_nodes', 'attention_ns', 'attention_share']) +
            '\nCorresponding raw timing ratios:\n\n' + md_table([p for p in full if p['seq'] in ('16', '64')],
                     ['model', 'seq', 'arm', 'l2_l1', 'l2_l05', 'l2_legacy']))
    section(15, 'Unmet gates and concrete next work',
            ('Full matrix measurements are present; inspect every FAIL above before final delivery. '
             'A performance failure requires a cell-specific diagnosis, which must be written after review.\n'
             if complete else 'Anchored searches/measurements remain unfinished; G-8 has no complete eight-cell verdict.\n') +
            '\nThe current long-running CPU path consists of `SearchContext::Evaluate` residency sweeps, '
            '`PrepareSymbolicProblem` task pricing and `ScheduleBySkeleton` lazy EST requeues. '
            'Candidate `.outer/*.result` files retain those phase costs. Further speed changes must '
            'preserve exact dependency sets, worker/slot/times, candidate coverage and acceptance order; '
            'the primary runs have not been restarted for unproven memoization speedups.\n\n'
            'Remaining acceptance work: finish all 32 anchored skeleton arms; driver-check each top-five; '
            'measure all shortlisted triples; audit each measured winner; recompute all gates; '
            'record measured failures with specific causes; finalize FINDINGS/TODO/STATUS and push.')
    exclusions = [
        ('TaskBody changes', [p for p in changed if re.search(r'TaskBody.*\.h$', p)]),
        ('CUTLASS submodule', [p for p in changed if p.startswith('third_party/cutlass')]),
        ('Old six placement heuristics', [p for p in changed if p in {
            'include/tilemega/Solver/JointPlacement.h', 'lib/Solver/EftPlacement.cpp',
            'lib/Solver/ChainPlacement.cpp', 'lib/Solver/BalancedPlacement.cpp',
            'lib/Solver/WavefrontPlacement.cpp', 'lib/Solver/PlanMaterialize.cpp'}]),
        ('Simulator implementation', [p for p in changed if p == 'lib/Solver/ExecutionSimulator.cpp']),
    ]
    section(16, 'Excluded scope', md_table([dict(item=k, changed=v or 'none') for k, v in exclusions], ['item', 'changed']) +
            '\nNo attention TaskBody rewrite, sync/barrier/event redesign, paging, static batch, '
            'tile-transfer fusion or correctness-criterion redesign is part of R9. '
            '`MIDPOINT_REFINE=0` is enforced by the raw build-command audit C-19/G-6. '
            'Compiler/front-end repairs and executor-order constraints are explicitly declared in section 5.')
    (E / 'summary.md').write_text('\n'.join(pieces))
    print(f'SUMMARY measured={len(perf)}/40 verify_exit={result.returncode} final_review=pending')


if __name__ == '__main__':
    main()
