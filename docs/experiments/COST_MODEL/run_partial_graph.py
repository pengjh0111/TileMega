#!/usr/bin/env python3
"""Validate measured float-partial calibration in fresh, frozen processes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--processes', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=41)
    args = parser.parse_args()
    if args.processes < 50 or args.batch < 1 or args.repeats < 1:
        raise ValueError('fresh-process validation requires at least 50 processes and a positive protocol')
    repo = Path(__file__).resolve().parents[3]
    binary = repo/'build-portable/tools/tilemega-calibrate'
    base = repo/'configs/targets/sm_89.json'
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    (output/'profiles').mkdir()
    manifest = dict(binary=str(binary), binary_sha256=sha(binary), base=str(base),
                    base_sha256=sha(base), batch=args.batch, repeats=args.repeats,
                    processes=args.processes,
                    commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip())
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'base.json').write_bytes(base.read_bytes())
    diff = subprocess.check_output(['git','diff','--binary'], cwd=repo)
    (output/'source.diff').write_bytes(diff)
    (output/'status.txt').write_text('RUNNING completed=0\n')
    fields = ['process','status','elapsed_s','fixed_ns','base_ns','d_l2_ns','d_dram_ns',
              'verified_elements','verified_chunks','log_sha256','profile_sha256']
    completed = 0
    try:
        with (output/'processes.tsv').open('w') as table:
            writer = csv.DictWriter(table, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            for index in range(args.processes):
                if sha(binary) != manifest['binary_sha256'] or sha(base) != manifest['base_sha256']:
                    raise RuntimeError('measurement binary or base changed during the frozen run')
                profile_path = output/'profiles'/f'{index:03d}.json'
                log_path = output/'logs'/f'{index:03d}.txt'
                command = [str(binary),'--dtype','bf16','--base',str(base),
                           '--fp32-partial-combine-only','--combine-graph-batch',str(args.batch),
                           '--repeats',str(args.repeats),'--out',str(profile_path)]
                start = time.monotonic()
                with log_path.open('w') as log:
                    result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=120)
                text = log_path.read_text()
                match = re.search(r'COMBINE_VERIFY elements=(\d+) chunks=(\d+) bits_equal=1',text)
                if result.returncode or not match:
                    raise RuntimeError(f'process {index}: exit={result.returncode}, output verification={bool(match)}')
                target = json.loads(profile_path.read_text())
                # Locate by schema field, then demand a single measured typed profile.
                profiles = []
                def visit(value):
                    if isinstance(value,dict):
                        if 'fp32_partial_combine' in value:
                            item = value['fp32_partial_combine']
                            if item.get('reason') == 'measured':
                                profiles.append(item)
                        for child in value.values():
                            visit(child)
                    elif isinstance(value,list):
                        for child in value:
                            visit(child)
                visit(target)
                if len(profiles) != 1:
                    raise RuntimeError('profile output does not identify one measured partial-storage profile')
                fit = profiles[0]
                row = dict(process=index,status='PASS',elapsed_s=time.monotonic()-start,
                           verified_elements=int(match[1]),verified_chunks=int(match[2]),
                           log_sha256=sha(log_path),profile_sha256=sha(profile_path))
                for key in ('fixed_ns','base_ns','d_l2_ns','d_dram_ns'):
                    row[key] = fit[key]
                writer.writerow(row)
                table.flush()
                completed += 1
                (output/'status.txt').write_text(f'RUNNING completed={completed}\n')
        (output/'status.txt').write_text(f'PASS completed={completed}/{args.processes}\n')
    except Exception as error:
        (output/'status.txt').write_text(f'STOP completed={completed}/{args.processes} reason={error}\n')
        raise


if __name__ == '__main__':
    main()
