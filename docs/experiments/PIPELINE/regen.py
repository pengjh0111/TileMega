#!/usr/bin/env python3
"""Re-project FORK6's six selected plans at HEAD so they carry the frontier.

The sources FORK6 measured were emitted before `no_producer` existed, so the
prefetch arms cannot be built from them: the field is guarded, and a build with
`TILEMEGA_PREFETCH_RUNTIME=1` against a source that omits it fails on
`-Werror=missing-field-initializers` rather than silently prefetching nothing.

Each cell is re-projected with FORK6's own recorded command -- same exporter,
same geometry, same kappa/residency, same driver source -- rebuilt against
HEAD.  `verify` then strips the guarded field back out of the result and
compares byte for byte against the file FORK6 measured; the cells are
comparable exactly to the extent that check passes.
"""
import argparse, json, pathlib, re, subprocess, sys, time

REPO = pathlib.Path(__file__).resolve().parents[3]
HERE = REPO / 'docs/experiments/PIPELINE'
FORK6 = REPO / 'docs/experiments/COSTMODEL/raw_kloop'
CELLS = ['gqa2_s4', 'gqa2_s128', 'mha4_s4', 'mha4_s128', 'real_s4', 'real_s128']
GUARD = re.compile(r'\n#if TILEMEGA_PREFETCH_RUNTIME\n   , (?:true|false)\n#endif\n  \},')


def fork6_source(cell):
    return pathlib.Path(json.loads((FORK6 / cell / 'specs.json').read_text())
                        ['selected']['source'])


def project(driver):
    for cell in CELLS:
        src = fork6_source(cell)
        cmd = json.loads((src.parent / 'process.json').read_text())['command']
        out = HERE / 'raw' / cell / 'src'
        out.mkdir(parents=True, exist_ok=True)
        # Only the driver and the destination move; every model argument is the
        # one FORK6 projected with.
        cmd = [str(driver)] + cmd[1:-2] + [str(out.resolve()), cmd[-1]]
        begin = time.time_ns()
        with (out / 'project.log').open('w') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
        (out / 'process.json').write_text(json.dumps(dict(
            command=cmd, exit_code=r.returncode, started_ns=begin,
            elapsed_ns=time.time_ns() - begin, fork6_source=str(src),
            placement=src.stem,
            head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=REPO, text=True).strip())) + '\n')
        print('REGEN', cell, src.stem, r.returncode, flush=True)


def vintage(cells, driver, scratch):
    """Re-project at a second driver vintage and compare every emitted file.

    The sources here were projected at `7d69529f9`; four commits landed after
    it that touch the solver's frontier and its price.  `no_producer` is
    emitted from `Frontend.cpp`'s write set, not from those, so the sources
    should be unaffected -- this checks that rather than arguing it.
    """
    rows = []
    for cell in cells:
        src = fork6_source(cell)
        old = HERE / 'raw' / cell / 'src'
        cmd = json.loads((old / 'process.json').read_text())['command']
        out = pathlib.Path(scratch) / cell
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(driver)] + cmd[1:-2] + [str(out.resolve()), cmd[-1]]
        with (out / 'project.log').open('w') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
        names = sorted(x.name for x in out.iterdir()
                       if x.name not in ('project.log',))
        differing = [n for n in names
                     if (old / n).read_bytes() != (out / n).read_bytes()]
        rows.append(dict(cell=cell, recorded_head=json.loads(
            (old / 'process.json').read_text())['head'],
            second_head=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            exit_code=r.returncode, files=len(names),
            differing=len(differing), differing_names=','.join(differing) or '-'))
        print('VINTAGE', cell, rows[-1]['differing'], flush=True)
    out = HERE / 'raw' / 'projection_vintage.tsv'
    with out.open('w') as f:
        f.write('\t'.join(rows[0]) + '\n')
        for r in rows:
            f.write('\t'.join(str(v) for v in r.values()) + '\n')
    return 1 if sum(r['differing'] for r in rows) else 0


def verify():
    rows, bad = [], 0
    for cell in CELLS:
        src = fork6_source(cell)
        new = HERE / 'raw' / cell / 'src' / src.name
        same = GUARD.sub('},', new.read_text()) == src.read_text()
        bad += not same
        rows.append(dict(cell=cell, placement=src.stem, fork6=str(src),
                         regenerated=str(new),
                         identical_without_frontier=int(same)))
    out = HERE / 'raw' / 'regeneration.tsv'
    with out.open('w') as f:
        f.write('\t'.join(rows[0]) + '\n')
        for r in rows:
            f.write('\t'.join(str(v) for v in r.values()) + '\n')
    print(f'REGEN_VERIFY cells={len(rows)} differing={bad}')
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['project', 'verify', 'vintage'])
    ap.add_argument('--driver', type=pathlib.Path, default='/tmp/r7-project')
    ap.add_argument('--cells', nargs='+', default=CELLS)
    ap.add_argument('--scratch', default='/tmp/r7-reproj')
    a = ap.parse_args()
    if a.action == 'project':
        project(a.driver)
        return 0
    if a.action == 'vintage':
        return vintage(a.cells, a.driver, a.scratch)
    return verify()


if __name__ == '__main__':
    sys.exit(main())
