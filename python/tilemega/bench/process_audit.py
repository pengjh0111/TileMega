"""Audit raw fresh-process evidence; summaries alone never certify a gate."""
import json
import re


def litmus_counts(raw):
    counts = {}
    for path in sorted((raw/'log').glob('*_a*_g*_t*.txt')):
        m = re.fullmatch(r'(per_writer|thread0_fence|no_barrier|no_fence)_a([01])_g(\d+)_t(\d+)\.txt',path.name)
        if not m:
            continue
        arm, acquire, grid, tile = m.groups()
        statuses = re.findall(r'^RESULT status=(\S+)',path.read_text(),re.M)
        counts[(arm,int(acquire),int(grid),int(tile))] = dict(
            runs=len(statuses),pass_count=statuses.count('pass'),
            mismatch=statuses.count('MISMATCH'),other=sum(x not in ('pass','MISMATCH') for x in statuses))
    return counts

def result(path, metadata=True):
    text = path.read_text()
    statuses = re.findall(r'^RESULT status=(\S+)', text, re.M)
    if len(statuses) != 1:
        raise ValueError(f'{path}: expected exactly one RESULT')
    meta = json.loads(path.with_suffix('.json').read_text()) if metadata else {}
    return statuses[0], meta

def correctness(raw, subset=False, chain=False, processes=5):
    if processes<5:raise ValueError("at least five fresh processes are required")
    details = []
    for model in ('gqa2', 'mha4'):
        cells = ((1, 0), (128, 512), (2048, 0)) if subset else ((4, 3), (128, 3))
        for seq, past in cells:
            folder = (raw/'correctness'/f'{model}_s{seq}'/'chain' if chain else
                      raw/('seqscan' if subset else 'correctness')/f'{model}_s{seq}_p{past}')
            if len(list(folder.glob('r*.log'))) != processes:
                raise ValueError(f'{folder}: expected {processes} fresh process logs')
            stamps = set()
            digests = set()
            for i in range(processes):
                status, meta = result(folder/f'r{i}.log')
                if status!='PASS' or meta['exit_code'] or meta['round']!=i:
                    raise ValueError(f'{folder}/r{i}: correctness failed')
                stamps.add(meta['time_ns'])
                digests.add(meta['binary_sha256'])
            if len(stamps)!=processes or len(digests)!=1:
                raise ValueError(f'{folder}: duplicated process timestamps or mixed binaries')
            key=f'{model}_s{seq}_chain' if chain else f'{model}_p0_full'
            build=json.loads((raw/'log'/f'{key}.build.json').read_text())
            if digests!={build['binary_sha256']} or build.get('exit_code',0):
                raise ValueError(f'{folder}: correctness executable differs from the compiled configuration')
            details.append(f'{model}/s{seq}/p{past}={processes}/{processes}')
    return '; '.join(details)
