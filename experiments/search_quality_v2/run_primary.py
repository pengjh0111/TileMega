"""Resume primary SQ2 after materialization and stop at the result gate."""

import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from calibration_queue import CASE
from guarded import PYTHON, ROOT, read, sha, write


SCRIPTS = Path(__file__).resolve().parent
PHASE = CASE / 'phase3'
RESULTS = ROOT / 'docs/results/sm90_search_quality_v2/qwen3_1_7b_b1'
PLOTSITE = '/root/TileMega-mechanism-ae/runs/sm90_mechanism_ae/plot-packages'


def execute(label, cmd, env=None):
    start = time.time()
    with (CASE / f'{label}.log').open('ab') as log:
        ret = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    write(CASE / f'{label}.phase.json', {'command': cmd, 'started': start, 'ended': time.time(), 'returncode': ret})
    if ret:
        raise RuntimeError(f'{label} failed with exit {ret}. See {CASE / (label + ".log")}')


def wait_materialization():
    while not (CASE / 'population-materialized.json').exists():
        pid = (PHASE / 'materialization.pid').read_text().strip()
        cmdline = Path('/proc') / pid / 'cmdline'
        if not cmdline.exists() or not cmdline.read_bytes():
            raise RuntimeError('materialization stopped. Inspect phase3/materialization-resumed.log')
        time.sleep(20)


def stage_compact_results():
    requested = [
        'report.md', 'deviations.md', 'numbers.json', 'paper_paragraph.txt', 'qspace.json',
        'calib_gemm.csv', 'calib_attention.csv', 'dev_rescore.csv', 'model_freeze.json',
        'frozen_search.json', 'population.json', 'candidates.csv', 'timings.csv', 'confirmation.csv',
        'selection_summary.csv', 'costs.json', 'analysis_population.csv', 'regret_curves.csv', 'plot.py',
    ]
    requested += [f'fig_{stem}.{ext}' for stem in ('SQ1', 'SQ2', 'SQ3', 'SQ4') for ext in ('png', 'pdf')]
    if (CASE / 'gpu_failover.json').exists():
        requested.append('gpu_failover.json')
    for entry in requested:
        source = CASE / entry
        assert source.is_file() and source.stat().st_size > 0, entry
        if entry == 'qspace.json':
            (RESULTS / entry).write_text(source.read_text().rstrip() + '\n')
        else:
            shutil.copyfile(source, RESULTS / entry)
    write(RESULTS / 'artifact_index.json', {
        'raw_run_directory': str(CASE),
        'artifacts': {entry: {'compact_sha256': sha(RESULTS / entry), 'source_sha256': sha(CASE / entry)}
                      for entry in requested},
    })
    write(CASE / 'results_staged.json', {'at': time.time(), 'path': str(RESULTS)})


def commit_compact_results():
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    assert git('branch', '--show-current') == 'sm90-search-quality-v2'
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('the index contains other work')
    path = str(RESULTS.relative_to(ROOT))
    git('add', '--', path)
    git('diff', '--cached', '--check')
    if git('diff', '--cached', '--name-only'):
        git('commit', '-m', 'docs: record SQ2 primary selection quality')
    write(CASE / 'publication.json', {'status': 'committed locally', 'head': git('rev-parse', 'HEAD'),
                                      'result_directory': path})


def main():
    with (CASE / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(CASE / 'campaign_pid.json', {'pid': os.getpid(), 'started': time.time()})
        wait_materialization()
        if not (CASE / 'frozen_search.json').exists():
            execute('freeze', [sys.executable, str(SCRIPTS / 'freeze_population.py')])
        if not (CASE / 'measurement_complete.json').exists():
            execute('test', [sys.executable, str(SCRIPTS / 'measure_population.py')])
        if not (CASE / 'numbers.json').exists():
            execute('analysis', [PYTHON, str(SCRIPTS / 'analyze.py')])
        if not (CASE / 'report.md').exists():
            execute('report', [sys.executable, str(SCRIPTS / 'write_report.py')])
        if not all((CASE / f'fig_{name}.{ext}').exists() for name in ('SQ1', 'SQ2', 'SQ3', 'SQ4')
                   for ext in ('pdf', 'png')):
            env = dict(os.environ)
            if Path(PLOTSITE).is_dir():
                env['PYTHONPATH'] = PLOTSITE
            execute('plot', [PYTHON, str(SCRIPTS / 'plot.py'), str(CASE)], env)
        stage_compact_results()
        criteria = read(CASE / 'numbers.json')['criteria']
        write(CASE / 'phase4_gate.json', {'criteria': criteria, 'all_passed': all(criteria.values()),
                                         'review_required_before_secondary': not all(criteria.values())})
        commit_compact_results()
        print('Primary measured, report generated. Criteria:', criteria, flush=True)


if __name__ == '__main__':
    main()
