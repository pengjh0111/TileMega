#!/usr/bin/env python3
"""B1-a's SEQSCAN subset: the twelve R5 cases, with the prefetch arms.

The plans JOINT measured were projected before `no_producer` existed, so a
prefetch build against them fails on the guarded field rather than quietly
prefetching nothing.  Each case is therefore re-projected here by replaying
JOINT's own recorded command and environment against a driver built at HEAD;
`verify` strips the guarded field back out and compares byte for byte with the
plan JOINT ran, which is what makes this the same subset and not a new one.

Only the two reference models are in the subset, as in R5 and R6.
"""
import argparse, concurrent.futures, json, os, pathlib, re, subprocess, sys, time

REPO = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'docs/experiments/JOINT'))
import measure

JOINT = REPO / 'docs/experiments/JOINT/raw'
CELLS = ['gqa2_s4', 'gqa2_s128', 'mha4_s4', 'mha4_s128']
CASES = ((1, 0), (128, 512), (2048, 0))
ARMS = ('prefetch', 'inline')
PAGE = 1024
GUARD = re.compile(r'\n#if TILEMEGA_PREFETCH_RUNTIME\n   , (?:true|false)\n#endif\n  \},')


def case_name(seq, past):
    return f's{seq}_p{past}'


def choice(cell):
    return json.loads((JOINT / cell / 'choice.json').read_text())['choice']


def project(cell, seq, past, driver):
    name = case_name(seq, past)
    old = json.loads((JOINT / cell / 'seqscan_plans' / name / 'process.json').read_text())
    out = HERE / 'raw/seqscan' / cell / name
    out.mkdir(parents=True, exist_ok=True)
    command = [str(driver)] + old['command'][1:-2] + [str(out.resolve()), old['command'][-1]]
    env = os.environ.copy()
    env.update(old['environment'])
    begin = time.time_ns()
    with (out / 'project.log').open('w') as f:
        r = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    (out / 'process.json').write_text(json.dumps(dict(
        command=command, environment=old['environment'], exit_code=r.returncode,
        started_ns=begin, elapsed_ns=time.time_ns() - begin,
        joint_plan=str(JOINT / cell / 'seqscan_plans' / name),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())) + '\n')
    print('SEQSCAN_PROJECT', cell, name, r.returncode, flush=True)
    return r.returncode


def verify():
    rows, bad = [], 0
    for cell in CELLS:
        placement = choice(cell)['placement']
        for seq, past in CASES:
            name = case_name(seq, past)
            old = JOINT / cell / 'seqscan_plans' / name / (placement + '.cu')
            new = HERE / 'raw/seqscan' / cell / name / (placement + '.cu')
            same = new.exists() and GUARD.sub('},', new.read_text()) == old.read_text()
            bad += not same
            rows.append(dict(cell=cell, case=name, placement=placement,
                             joint=str(old), regenerated=str(new),
                             identical_without_frontier=int(same)))
    out = HERE / 'raw/seqscan/regeneration.tsv'
    with out.open('w') as f:
        f.write('\t'.join(rows[0]) + '\n')
        for r in rows:
            f.write('\t'.join(str(v) for v in r.values()) + '\n')
    print(f'SEQSCAN_VERIFY cases={len(rows)} differing={bad}')
    return 1 if bad else 0


def spec(cell, seq, past, arm):
    chosen = choice(cell)
    source = HERE / 'raw/seqscan' / cell / case_name(seq, past) / (chosen['placement'] + '.cu')
    extra = ['PREFETCH_RUNTIME=1'] + (['PREFETCH_INLINE=1'] if arm == 'inline' else [])
    if PAGE != 1024:
        extra.append(f'PREFETCH_PAGE_BYTES={PAGE}')
    return dict(chosen, source=str(source), placement_macro='0', extra=extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['project', 'verify', 'build', 'run'])
    ap.add_argument('--driver', type=pathlib.Path, default=pathlib.Path('/tmp/r7-project'))
    ap.add_argument('--cells', nargs='+', default=CELLS)
    ap.add_argument('--rounds', type=int, default=50)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--arch', default='sm_89')
    a = ap.parse_args()
    if a.action == 'verify':
        return verify()
    if a.action == 'project':
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            jobs = [pool.submit(project, c, seq, past, a.driver)
                    for c in a.cells for seq, past in CASES]
            return 1 if sum(j.result() != 0 for j in jobs) else 0
    if a.action == 'build':
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            jobs = [pool.submit(measure.build, HERE / 'raw/seqscan' / c, c.split('_')[0],
                                f'{case_name(seq, past)}_{arm}', spec(c, seq, past, arm), a.arch)
                    for c in a.cells for seq, past in CASES for arm in ARMS]
            return 1 if sum(j.result() != 0 for j in jobs) else 0
    session = str(time.time_ns())
    for c in a.cells:
        model = c.split('_')[0]
        for seq, past in CASES:
            for arm in ARMS:
                folder = HERE / 'raw/seqscan' / c / 'seqscan' / case_name(seq, past) / arm
                for i in range(a.rounds):
                    measure.run(HERE / 'raw/seqscan' / c, model, seq,
                                f'{case_name(seq, past)}_{arm}', folder, i, 0, session, past=past)
                print('SEQSCAN', c, case_name(seq, past), arm, f'{a.rounds}/{a.rounds}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
