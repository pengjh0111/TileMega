#!/usr/bin/env python3
"""Raw-evidence readers shared by the final verifier and analysis scripts."""
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def result(path, metadata=True):
    text = path.read_text()
    statuses = re.findall(r'^RESULT status=(\S+)', text, re.M)
    if len(statuses) != 1:
        raise ValueError(f'{path}: expected exactly one RESULT')
    meta = json.loads(path.with_suffix('.json').read_text()) if metadata else {}
    return statuses[0], meta


def litmus(raw=None):
    raw = raw or HERE / 'litmus_v3'
    build = json.loads((raw / 'build.json').read_text())
    if hashlib.sha256((HERE / 'litmus.cu').read_bytes()).hexdigest() != build['source_sha256']:
        raise ValueError('litmus source differs from the compiled experiment')
    details = []
    for suite, negative in (('cache', 'no_fence'), ('skew', 'no_barrier')):
        for grid in (64, 128, 256):
            for tile in (1024, 4096):
                for arm in ('per_writer', 'thread0_fence', negative):
                    folder = raw / 'scan' / suite / f'g{grid}_t{tile}' / arm
                    paths = sorted(folder.glob('r*.log'))
                    if len(paths) != 50:
                        raise ValueError(f'{folder}: {len(paths)}/50 raw runs')
                    passed = 0
                    expected = 'MISMATCH' if arm == negative else 'pass'
                    for i in range(50):
                        status, meta = result(folder / f'r{i}.log')
                        command = meta['command']
                        skew = command[command.index('--writer-skew-cycles')+1]
                        if skew != ('2000000' if suite == 'skew' else '0'):
                            raise ValueError(f'{folder}: stress changed between arms')
                        if ('--no-acquire-fence' in command) != (suite == 'cache'):
                            raise ValueError(f'{folder}: inconsistent consumer acquire')
                        if command[command.index('--release')+1] != arm or meta['round'] != i:
                            raise ValueError(f'{folder}: mislabeled process')
                        passed += status == expected and meta['exit_code'] == (1 if arm == negative else 0)
                    details.append(f'{suite}/g{grid}/t{tile}/{arm}={passed}/50')
                    if passed != 50:
                        raise ValueError(details[-1])
    return '; '.join(details)


def five_arm(raw):
    details = []
    sessions = set()
    for model in ('gqa2', 'mha4'):
        for seq in (4, 128):
            for placement in (0, 5):
                for arm in ('full', 'nofence', 'nowait', 'neither', 'l1nosync'):
                    folder = raw / 'paired' / f'{model}_s{seq}_p{placement}' / arm
                    if len(list(folder.glob('r*.log'))) != 25:
                        raise ValueError(f'{folder}: not 25 logs')
                    for i in range(25):
                        path = folder / f'r{i}.log'
                        status, meta = result(path)
                        if arm == 'full' and (status != 'PASS' or meta['exit_code']):
                            raise ValueError(f'{path}: full correctness failed')
                        if meta['exit_code'] not in (0, 1):
                            raise ValueError(f'{path}: process did not complete normally')
                        if len(re.findall(r'^E2E_TIME ', path.read_text(), re.M)) != 1:
                            raise ValueError(f'{path}: missing timing')
                        combo = (0 if placement == 0 else 5) + ('full', 'nofence', 'nowait', 'neither', 'l1nosync').index(arm)
                        if (i + meta['order']) % 10 != combo or meta['round'] != i:
                            raise ValueError(f'{path}: arm order is not rotated')
                        sessions.add(meta['session'])
                    details.append(f'{model}/s{seq}/p{placement}/{arm}=25/25')
    if len(sessions) != 1:
        raise ValueError(f'mixed measurement sessions: {sessions}')
    return '; '.join(details)


def targets(analyzer):
    frozen = json.loads((HERE / 'targets_raw/frozen.json').read_text())
    target = HERE / 'targets.tsv'
    if hashlib.sha256(target.read_bytes()).hexdigest() != frozen['target_sha256']:
        raise ValueError('frozen targets changed')
    rows = list(csv.DictReader(target.open(), delimiter='\t'))
    expected = {(c, m, str(s)) for c in ('legacy_grid_stride', 'balanced', 'rotate', 'eft', 'wavefront', 'chain')
                for m in ('gqa2', 'mha4') for s in (4, 128)}
    if len(rows) != 24 or {(r['candidate'], r['model'], r['seq']) for r in rows} != expected:
        raise ValueError('targets do not cover exactly six candidates x four cells')
    for row in rows:
        r = analyzer.analyze(REPO / row['dump'], REPO / row['source'], 1)
        floor = max(r['cp_corrected_ns'], r['queue_lb_ns']) / 1e6
        measured = r['measured_l2_ms']
        ceiling = (floor + measured) / 2
        if floor > measured or any(abs(float(row[k])-v) > 1e-9 for k, v in
                                  (('floor_ms', floor), ('measured_A_ms', measured), ('target_ms', ceiling))):
            raise ValueError(f'{row["dump"]}: target does not reproduce from raw dump')
    for p, digest in frozen['inputs'].items():
        if hashlib.sha256((REPO / p).read_bytes()).hexdigest() != digest:
            raise ValueError(f'frozen input modified: {p}')
    return '24/24 candidate-specific targets reproduce; 0 targets below their own floor'
