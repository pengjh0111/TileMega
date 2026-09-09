#!/usr/bin/env python3
"""A9 twelve-cell steady-state calibration; reused four-arm measurement path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arch', default='native')
    p.add_argument('--phases', default='prepare,build,correctness,attrib')
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
                if phase in requested and (a.out/f'{phase}.tsv').exists():
                    raise RuntimeError(f'refusing to overwrite {phase} evidence')
            command = [sys.executable,str(repo/'docs/experiments/L2_ATTRIB/run_t1.py'),
                       '--out',str(a.out),'--arch',a.arch,'--variants','base',
                       '--seqs','1,4,16,128,512,2048','--kappa','1',
                       '--correctness-runs','50','--runs','25',
                       '--phases',','.join(requested)]
            (a.out/'calibration_command.json').write_text(json.dumps(dict(
                command=command,warmup=5,repeat=11,correctness_processes=600,
                attribution_processes=1200,models=['gqa2','mha4'],dtype='bf16'),indent=2)+'\n')
            subprocess.run(command,check=True,env=env)
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
