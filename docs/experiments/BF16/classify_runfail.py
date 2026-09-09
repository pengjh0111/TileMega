#!/usr/bin/env python3
"""Replay the 308 *historical* failing binaries; no performance inference."""
import csv
import hashlib
from pathlib import Path
import subprocess
from collections import Counter

repo = Path(__file__).resolve().parents[3]
original = repo / 'docs/experiments/ORACLE/raw_bf16'
out = Path(__file__).resolve().parent / 'runfail_audit'
(out / 'logs').mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((original / 'screen_mha4.tsv').open(), delimiter='\t'))
failed = [row for row in rows if row['status'] == 'RUNFAIL']
if len(failed) != 308:
    raise RuntimeError(f'input changed: expected 308 RUNFAIL, got {len(failed)}')
counts = Counter()
with (out / 'classification.tsv').open('w') as handle:
    writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
    writer.writerow(['config', 'historical_status', 'replay_class', 'exit_code', 'binary_sha256', 'log'])
    for row in failed:
        tag = (f"mha4_{row['tile_m']}x{row['tile_n']}x{row['tile_k']}"
               f"s{row['stages']}k{row['split_k']}")
        binary = original / 'bin' / tag
        digest = hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else '-'
        code, output = -1, ''
        if not binary.exists():
            kind = 'missing_binary'
        else:
            try:
                result = subprocess.run([str(binary), str(original / 'export/mha4/fixture')],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, timeout=120)
                code, output = result.returncode, result.stdout
                if 'RESULT status=PASS' in output and code == 0:
                    kind = 'not_reproduced'
                elif 'E2E_DIFF ' in output and ('status=FAIL' in output or 'status=MISMATCH' in output):
                    kind = 'numerical_criterion'
                elif 'too many resources' in output or 'out of memory' in output:
                    kind = 'resource_error'
                elif 'illegal memory access' in output or 'invalid configuration' in output:
                    kind = 'cuda_runtime_error'
                else:
                    kind = 'unclassified'
            except subprocess.TimeoutExpired as error:
                kind, code = 'timeout', 124
                output = error.stdout or b''
                if isinstance(output, bytes): output = output.decode(errors='replace')
        log = out / 'logs' / f'{tag}.txt'
        log.write_text(output)
        writer.writerow([tag, 'RUNFAIL', kind, code, digest, str(log.relative_to(out))])
        handle.flush()
        counts[kind] += 1
        print(tag, kind, flush=True)
with (out / 'counts.tsv').open('w') as handle:
    writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
    writer.writerow(['replay_class', 'configurations'])
    writer.writerows(sorted(counts.items()))
