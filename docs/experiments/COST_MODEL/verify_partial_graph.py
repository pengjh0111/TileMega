#!/usr/bin/env python3
"""Independently audit frozen process logs and summarize measured coefficients."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', default='verification.json')
    args = parser.parse_args()
    directory = args.directory
    output = directory/args.output
    if output.exists():
        raise RuntimeError('refusing to overwrite verified evidence')
    manifest = json.loads((directory/'manifest.json').read_text())
    if sha(directory/'base.json') != manifest['base_sha256']:
        raise RuntimeError('frozen base changed')
    rows = list(csv.DictReader((directory/'processes.tsv').open(), delimiter='\t'))
    if len(rows) != manifest['processes'] or len(rows) < 50:
        raise RuntimeError('incomplete fresh-process matrix')
    summary = dict(processes=len(rows), verified_elements=0, verified_chunk_contributions=0,
                   rates={}, source_sha256=sha(directory/'source.diff'))
    for index,row in enumerate(rows):
        if int(row['process']) != index or row['status'] != 'PASS':
            raise RuntimeError('duplicate/missing/failed process')
        log = directory/'logs'/f'{index:03d}.txt'
        profile = directory/'profiles'/f'{index:03d}.json'
        if sha(log) != row['log_sha256'] or sha(profile) != row['profile_sha256']:
            raise RuntimeError('raw log/profile hash mismatch')
        target = json.loads(profile.read_text())
        fits = []
        def visit(value):
            if isinstance(value, dict):
                fit = value.get('fp32_partial_combine')
                if fit and fit.get('reason') == 'measured':
                    fits.append(fit)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(target)
        if len(fits) != 1 or any(fits[0][key] != float(row[key])
                                for key in ('fixed_ns','base_ns','d_l2_ns','d_dram_ns')):
            raise RuntimeError('process table does not match the typed profile')
        text = log.read_text()
        match = re.search(r'COMBINE_VERIFY elements=(\d+) chunks=(\d+) bits_equal=1',text)
        if not match or int(match[1]) != int(row['verified_elements']) or int(match[2]) != int(row['verified_chunks']):
            raise RuntimeError('CPU output verification was not observed')
        signals = re.findall(r'COMBINE_FIXED_RESOLUTION per_launch_ns=(\S+) launches_per_interval=(\d+) '
                             r'interval_signal_ns=(\S+) required_interval_ns=(\S+) resolved=1',text)
        if len(signals) != 1 or not float(signals[0][2]) > float(signals[0][3]):
            raise RuntimeError('raw timed interval does not resolve the fixed signal')
        arms = re.findall(r'COMBINE_PAIR round=(\d+) batch=(\d+) first=(\d+)',text)
        if not arms or any(int(batch) != manifest['batch'] or int(first) != int(round_) % 2
                           for round_,batch,first in arms):
            raise RuntimeError('paired graph arms were not interleaved')
        summary['verified_elements'] += int(match[1])
        summary['verified_chunk_contributions'] += int(match[1])*int(match[2])
    rng = np.random.default_rng(0)
    for field in ('fixed_ns','base_ns','d_l2_ns','d_dram_ns'):
        values = np.array([float(row[field]) for row in rows])
        if not np.all(np.isfinite(values)) or not np.all(values > 0):
            raise RuntimeError('unresolved measured coefficient')
        estimates = np.median(values[rng.integers(0,len(values),size=(10000,len(values)))],axis=1)
        summary['rates'][field] = dict(median=float(np.median(values)),
            minimum=float(values.min()),maximum=float(values.max()),
            median_ci95=[float(x) for x in np.quantile(estimates,[.025,.975])])
    summary['status'] = 'PASS'
    output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
