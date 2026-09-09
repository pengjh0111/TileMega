#!/usr/bin/env python3
"""A9 twelve-cell steady-state calibration; reused four-arm measurement path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def verify_artifacts(out, repo):
    recorded = json.loads((out/'completed_artifacts.json').read_text())
    binaries = {str(path.resolve()) for path in out.glob('bin/*') if path.is_file()}
    expected = set()
    for name, digest in recorded.items():
        path = Path(name)
        if not path.is_absolute():
            path = repo/path
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'calibration artifact changed: {path}')
        if path.parent.resolve() == (out/'bin').resolve():
            expected.add(str(path.resolve()))
    if len(expected) != 8 or binaries != expected:
        raise RuntimeError('calibration requires exactly the eight recorded four-arm binaries')
    return recorded


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arch', default='native')
    p.add_argument('--phases', default='prepare,build,correctness,attrib')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[3]
    phases = a.phases.split(',')
    if not set(phases) <= {'prepare','build','correctness','attrib'}:
        p.error('unknown phase')
    a.out.mkdir(parents=True, exist_ok=True)
    status = a.out/'status.txt'
    env = dict(os.environ, TILEMEGA_WARMUP='5', TILEMEGA_REPEAT='11')
    seqscan = repo/'docs/experiments/SEQSCAN/raw'
    try:
        status.write_text('RUNNING; no acceptance until all requested phases complete\n')
        if 'prepare' in phases:
            # Reuse the exact exported programs and fixture writers, not new
            # models or numerical references. Existing fixtures are immutable.
            for model in ('gqa2','mha4'):
                for seq in (1,4,16,128,512,2048):
                    dest = seqscan/f'fixture/{model}_s{seq}_p3'
                    if dest.exists():
                        if not (dest/'manifest.json').is_file():
                            raise RuntimeError(f'incomplete existing fixture: {dest}')
                        continue
                    if model == 'gqa2':
                        command = [sys.executable, str(repo/'docs/experiments/E2E/prepare_e2e.py'),
                                   '--vh-raw',str(seqscan/'export/gqa2')]
                    else:
                        command = [sys.executable,str(repo/'docs/experiments/P3_GENERALIZATION/prepare_fixture.py'),
                                   '--repo',str(repo),'--program',
                                   str(seqscan/'export/mha4/exported_program.pt2')]
                    command += ['--out',str(dest),'--seq',str(seq),'--past','3']
                    with (a.out/f'prepare_{model}_s{seq}.txt').open('w') as log:
                        subprocess.run(command,stdout=log,stderr=log,check=True,env=env)
        requested = [phase for phase in phases if phase != 'prepare']
        if requested:
            for model in ('gqa2','mha4'):
                for seq in (1,4,16,128,512,2048):
                    fixture = seqscan/f'fixture/{model}_s{seq}_p3/manifest.json'
                    meta = json.loads(fixture.read_text())
                    dims = meta.get('config',meta)
                    if (dims.get('seq'),dims.get('past'),dims.get('dtype')) != (seq,3,'torch.bfloat16'):
                        raise RuntimeError(f'fixture dimensions or dtype differ: {fixture}')
            for phase in ('correctness','attrib'):
                if phase in requested and (a.out/f'{phase}.tsv').exists() and not a.resume:
                    raise RuntimeError(f'refusing to overwrite {phase} evidence')
            if 'build' not in requested:
                # A run-only continuation must use the frozen build, not bless
                # whatever executable happens to be present after measurement.
                recorded = verify_artifacts(a.out,repo)
                (a.out/'verified_before_run.json').write_text(json.dumps(recorded,indent=2)+'\n')
            command = [sys.executable,str(repo/'docs/experiments/L2_ATTRIB/run_t1.py'),
                       '--out',str(a.out),'--arch',a.arch,'--variants','base',
                       '--seqs','1,4,16,128,512,2048','--kappa','1',
                       '--correctness-runs','50','--runs','25',
                       '--phases',','.join(requested)]
            if a.resume:
                if set(requested) - {'correctness','attrib'}:
                    raise RuntimeError('resume is restricted to frozen measurement phases')
                snapshot = a.out/'resume_prefix.json'
                if snapshot.exists():
                    raise RuntimeError('inspect the previous resume before another continuation')
                snapshot.write_text(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [a.out/'calibration_command.json',a.out/'attrib.tsv',a.out/'correctness.tsv']
                    if path.exists()},indent=2)+'\n')
                command += ['--resume']
            (a.out/'calibration_command.json').write_text(json.dumps(dict(
                command=command,warmup=5,repeat=11,correctness_processes=600,
                attribution_processes=1200,models=['gqa2','mha4'],dtype='bf16'),indent=2)+'\n')
            subprocess.run(command,check=True,env=env)
            if 'build' not in requested:
                verify_artifacts(a.out,repo)
            inputs = [*a.out.glob('bin/*'),*a.out.glob('ptxas/*'),
                      repo/'include/tilemega/Codegen/RuntimeOwnership.h',
                      repo/'configs/targets/sm_89.json', repo/'configs/targets/sm_120.json']
            (a.out/'completed_artifacts.json').write_text(json.dumps(
                {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},indent=2)+'\n')
        status.write_text('REQUESTED PHASES COMPLETE; A9 fit and functional gate remain separate\n')
    except BaseException:
        status.write_text('STOPPED; inspect logs; other independent prompt items remain active\n')
        raise


if __name__ == '__main__':
    main()
