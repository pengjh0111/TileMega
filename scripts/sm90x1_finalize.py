#!/usr/bin/env python3
"""Commit compact SM90X1 results after the guarded campaign becomes quiescent."""
import argparse
from pathlib import Path
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, write
from sm90x1 import RUN
from sm90x1_campaign import live_start

FILES = ['docs/results/SM90X1_'+name for name in (
    'phase0.csv', 'candidates.csv', 'stage_deltas.csv', 'arms.csv', 'summary.csv',
    'figure_arms.csv', 'figure_scatter.csv', 'h800.md',
    'uncertified_arms.csv', 'summary_user_waived.csv')]
TERMINAL = {'complete', 'finished_with_failures', 'stopped', 'blocked'}


def finish(state):
    k=state.get('k', read(RUN/'pilot-gate.json')['k'] if (RUN/'pilot-gate.json').exists() else 4)
    subprocess.run([PYTHON, str(ROOT/'scripts/sm90x1.py'), 'tables'], cwd=ROOT, check=False)
    subprocess.check_call([PYTHON, str(ROOT/'scripts/sm90x1.py'), 'tables', '--phase', '1',
                           '--cell', 'all', '--k', str(k)], cwd=ROOT)
    # The finalizer owns only generated result files. Human/source edits and a
    # pre-existing index require a separate review; never include them implicitly.
    subprocess.check_call(['git', 'diff', '--cached', '--quiet'], cwd=ROOT)
    changed=set(subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines())
    if changed-set(FILES):
        raise ValueError('unrelated tracked edits remain: '+str(sorted(changed-set(FILES))))
    subprocess.check_call(['git','diff','--check'],cwd=ROOT)
    if changed:
        subprocess.check_call(['git','add','--',*FILES],cwd=ROOT)
        subprocess.check_call(['git','commit','-m','docs: record SM90X1 campaign results'],cwd=ROOT)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return dict(status='finalized',campaign=state,result_commit=commit,at=time.time())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true',help='return if the campaign is still active')
    args=parser.parse_args()
    while True:
        state=read(RUN/'campaign-state.json')
        if state['status'] in TERMINAL:break
        if args.once:return 2
        campaign=read(RUN/'campaign-process.json')
        if live_start(campaign['pid']) is None:
            raise RuntimeError('campaign ended without a terminal state; retain results for review')
        time.sleep(30)
    write(RUN/'finalizer-state.json',finish(state))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        write(RUN/'finalizer-state.json',dict(status='failed',error=str(error),at=time.time()))
        raise
