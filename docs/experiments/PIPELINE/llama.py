#!/usr/bin/env python3
"""B1-a's Llama gate: R6's admitted maximal connected graph, with the arms.

The graph, the geometry, the fixture, the seed and the 0.0231875014 tolerance
are R6's admission run (`MODELS/covered_llama_admitted2`).  Only two things
change: the solve command is replayed against HEAD so the source carries the
buffer frontier the prefetch build needs, and the binaries add the mechanism.
The replay is a fresh solve, and it does not land on R6's geometry: R6's source
was recorded 65 commits before R7's baseline and the search now draws `split2`
where it drew `split4`.  `verify` records that difference instead of hiding it,
`sigma_off` re-solves with the pipelining dimension priced at zero, and
`sigma_page` re-solves with the solver's page set to the one the arms are built
with.  Both controls come back byte-identical to the default solve, so neither
this round's pricing nor the page is what moved the geometry (F-228).

The model's hidden size is 2048, which would make an RMSNorm scale row 4096
bytes -- but this graph has no normalization stage at all, because R6 admitted
it with both per-layer norms passed in as graph inputs.  `E2E_PREFETCH` reports
`declared=0`, and the run is a correctness gate on a graph the mechanism cannot
reach rather than a measurement of it.
"""
import argparse, concurrent.futures, fcntl, hashlib, json, os, pathlib, re, subprocess, sys, time

REPO = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'docs/experiments/JOINT'))
import measure

R6 = REPO / 'docs/experiments/MODELS/covered_llama_admitted2'
ROOT = HERE / 'raw/llama'
FIXTURE = REPO / 'docs/experiments/MODELS/covered_llama/fixture'
ARMS = ('control', 'prefetch', 'inline')
PAGE = 4096
EXTRA = ['NORM_EPSILON=1e-5f', 'ROPE_FP32_PHASE=1', 'MIDPOINT_REFINE=1']
GUARD = re.compile(r'\n#if TILEMEGA_PREFETCH_RUNTIME\n   , (?:true|false)\n#endif\n  \},')
# R6's recorded invocation (`JOINT2/summary.md`, the admitted2 line): capacity 3
# and the numerical rejection list are what made this geometry the admitted one.
COMMAND = ['build-portable/tools/tilemega-compile',
           'docs/experiments/MODELS/covered_llama/exported_program.pt2',
           '{out}/auto.cu', '--solve', 'docs/experiments/COSTMODEL/event_fit/target.json',
           '--seq', '4', '--past', '3', '--search-capacity', '3',
           '--search-domain', 'docs/experiments/COSTMODEL/event_fit/search_domain.json',
           '--dump-cg', '{out}/auto.mlir',
           '--hop-curve', 'docs/experiments/SIMULATOR/hop_ns.tsv',
           '--numerical-rejections', 'docs/experiments/MODELS/covered_llama/numerical_rejections.json']


def solve(out=ROOT, page=None):
    out.mkdir(parents=True, exist_ok=True)
    command = [str(REPO / x.format(out=str(out.resolve()))) if not x.startswith('-') and '/' in x
               else x.format(out=str(out.resolve())) for x in COMMAND]
    if page is not None:
        command += ['--prefetch-page-bytes', str(page)]
    begin = time.time_ns()
    with (out / 'solve.log').open('w') as f:
        r = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    (out / 'solve.json').write_text(json.dumps(dict(
        command=command, exit_code=r.returncode, started_ns=begin,
        elapsed_ns=time.time_ns() - begin, r6_source=str(R6 / 'auto.cu'),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()),
        indent=2) + '\n')
    print('LLAMA_SOLVE', out.name, r.returncode, flush=True)
    return r.returncode


def winner(folder):
    """The first ranked row of a solve's shortlist, as the solve wrote it."""
    rows = (folder / 'auto.cu.top3.tsv').read_text().splitlines()
    head = rows[0].split('\t')
    return dict(zip(head, rows[1].split('\t')))


def verify():
    """Report how the replayed solve differs from the source R6 admitted.

    R6's file predates `no_producer`, so identity was the hope rather than the
    premise; where the two differ the difference is the result.  `sigma_off`,
    if it has been run, separates this round's pricing from the rounds between.
    """
    same = GUARD.sub('},', (ROOT / 'auto.cu').read_text()) == (R6 / 'auto.cu').read_text()
    rows = [('covered_llama_admitted2', R6), ('replayed_at_head', ROOT)]
    for name, folder in (('replayed_without_sigma_pipeline', 'llama_sigma0'),
                         (f'replayed_with_sigma_page_{PAGE}', 'llama_sigma_page')):
        if (ROOT.parent / folder / 'auto.cu.top3.tsv').exists():
            rows.append((name, ROOT.parent / folder))
    columns = ('key', 'split_k', 'kappa', 'residency', 'floor_ns', 'predicted_ns')
    text = ['solve\tsource\tidentical_without_frontier\t' + '\t'.join(columns)]
    for name, folder in rows:
        w = winner(folder)
        text.append('\t'.join([name, str(folder / 'auto.cu'),
                               str(int(same)) if folder == ROOT else '-',
                               *(w[c] for c in columns)]))
    (ROOT / 'regeneration.tsv').write_text('\n'.join(text) + '\n')
    print(f'LLAMA_VERIFY identical_without_frontier={int(same)}')
    for line in text:
        print(line)
    return 0


def spec(arm):
    extra = list(EXTRA)
    if arm != 'control':
        extra += ['PREFETCH_RUNTIME=1', f'PREFETCH_PAGE_BYTES={PAGE}']
        if arm == 'inline':
            extra.append('PREFETCH_INLINE=1')
    return dict(source=str(ROOT / 'auto.cu'), kappa='1', residency='3',
                placement_macro='0', extra=extra)


def run(arm, rounds):
    binary = ROOT / 'bin' / arm
    folder = ROOT / 'correctness' / arm
    folder.mkdir(parents=True, exist_ok=True)
    session = str(time.time_ns())
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    for i in range(rounds):
        log = folder / f'r{i}.log'
        if log.exists():
            raise RuntimeError('refusing overwrite ' + str(log))
        env = {k: v for k, v in os.environ.items() if not k.startswith('TILEMEGA_')}
        env.update(TILEMEGA_WARMUP='0', TILEMEGA_REPEAT='1')
        command = [str(binary.resolve()), str(FIXTURE.resolve())]
        with open('/tmp/tilemega-r5-gpu.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            begin = time.time_ns()
            r = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1800)
        log.write_text(r.stdout + r.stderr)
        log.with_suffix('.json').write_text(json.dumps(dict(
            command=command, round=i, session=session, defines=spec(arm)['extra'],
            started_ns=begin, elapsed_ns=time.time_ns() - begin,
            exit_code=r.returncode, binary_sha256=digest)) + '\n')
        if r.returncode or 'RESULT status=PASS' not in r.stdout:
            raise RuntimeError(str(log))
        print('LLAMA', arm, i + 1, '/', rounds, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['solve', 'sigma_off', 'sigma_page', 'verify',
                                       'build', 'run'])
    ap.add_argument('--arms', nargs='+', default=['prefetch', 'inline'])
    ap.add_argument('--rounds', type=int, default=50)
    ap.add_argument('--arch', default='sm_89')
    a = ap.parse_args()
    if a.action == 'solve':
        return solve()
    if a.action == 'sigma_off':
        return solve(ROOT.parent / 'llama_sigma0', page=0)
    if a.action == 'sigma_page':
        return solve(ROOT.parent / 'llama_sigma_page', page=PAGE)
    if a.action == 'verify':
        return verify()
    if a.action == 'build':
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.arms)) as pool:
            jobs = [pool.submit(measure.build, ROOT, 'llama', arm, spec(arm), a.arch)
                    for arm in a.arms]
            return 1 if sum(j.result() != 0 for j in jobs) else 0
    for arm in a.arms:
        run(arm, a.rounds)
    return 0


if __name__ == '__main__':
    sys.exit(main())
