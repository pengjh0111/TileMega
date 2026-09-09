#!/usr/bin/env python3
"""A2 exact CPU-vs-archived-runtime counters; no GPU correctness claim."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time


def records(path, prefix):
    return [dict(re.findall(r'(\w+)=([^\s]+)', line))
            for line in path.read_text().splitlines() if line.startswith(prefix+' ')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gqa2','mha4'])
    parser.add_argument('--splits', nargs='+', type=int, default=[1,2,4,8,16])
    parser.add_argument('--pasts', nargs='+', type=int, default=[0,3,512])
    parser.add_argument('--order', type=int, choices=[0,1], default=1)
    parser.add_argument('--partition-workers', type=int, choices=[0,1], default=0)
    parser.add_argument('--split-periods', type=int, choices=[0,1], default=0)
    parser.add_argument('--tool',type=Path)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of completed fresh-process logs to compare per cell')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--runtime-logs', type=Path,
                        help='independent one-process-per-cell captures; not a synchronization gate')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    out = args.out or here/'runtime_projection/matrix'
    out.mkdir(parents=True, exist_ok=True)
    tool = args.tool or repo/'build-portable/tools/tilemega-runtime-projection'
    manifest = dict(arguments={**vars(args),'out':str(out),'runtime_logs':str(args.runtime_logs),'tool':str(tool)}, tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
                    commands=[], archive_comparisons=0, unverified_cells=[])
    (out/'status.txt').write_text('RUNNING CPU projection audit; no GPU launches\n')
    try:
        with (out/'counts.tsv').open('w') as stream:
            fields = ['model','seq','past','split','kappa','stages','task_refs','waits',
                      'max_worker_tasks','archive_processes','comparison']
            writer = csv.DictWriter(stream,fields,delimiter='\t',lineterminator='\n')
            writer.writeheader()
            for model in args.models:
                for split in args.splits:
                    source = repo/f'docs/experiments/BF16/raw_splitk/logs/{model}_k{split}_fp321_r0.txt'
                    resource = records(source,'E2E_RESOURCE')[0]
                    plan_path = repo/f'docs/experiments/BF16/raw_splitk/plan/{model}_k{split}.json'
                    plan = json.loads(plan_path.read_text())['variants'][0]
                    if not all(plan.get(k) for k in ('rope_tile_per_block','kv_tile_per_block',
                                                       'activation_tile_per_block','combiner_tile_per_block')):
                        raise RuntimeError('archive does not use uniform tile ownership')
                    tile = plan['uniform']
                    requests = ['symbolic'] if len(args.pasts)>1 else [str(args.pasts[0])]
                    for request in requests:
                        if hashlib.sha256(tool.read_bytes()).hexdigest() != manifest['tool_sha256']:
                            raise RuntimeError('projection binary changed during the experiment')
                        command = [str(tool),str(repo/f'docs/experiments/SEQSCAN/raw/export/{model}.json'),
                                   resource['grid'],resource['block'],'1',request,
                                   *[str(tile[k]) for k in ('tile_m','tile_n','tile_k','stages','split_k')],
                                   'tile',str(args.order),str(args.partition_workers),str(args.split_periods)]
                        start = time.monotonic()
                        tag = f'{model}_k{split}_p{request}'
                        try:
                            run = subprocess.run(command,capture_output=True,text=True,timeout=600,
                                                 env=dict(os.environ,TILEMEGA_ISL_AUDIT='1'))
                        except subprocess.TimeoutExpired as error:
                            decode = lambda value: value.decode(errors='replace') if isinstance(value,bytes) else value or ''
                            (out/f'{tag}.tsv').write_text(decode(error.stdout))
                            (out/f'{tag}.log').write_text(decode(error.stderr))
                            manifest['commands'].append(dict(command=command,seconds=time.monotonic()-start,
                                                             result='TIMEOUT',returncode=None))
                            raise
                        (out/f'{tag}.tsv').write_text(run.stdout)
                        (out/f'{tag}.log').write_text(run.stderr)
                        manifest['commands'].append(dict(command=command,seconds=time.monotonic()-start,
                                                         returncode=run.returncode))
                        run.check_returncode()
                        if 'ISL_CONTEXT remaining=0' not in run.stderr:
                            raise RuntimeError('missing zero-reference evidence')
                        for row in csv.DictReader(io.StringIO(run.stdout),delimiter='\t'):
                            seq = int(row['seq'])
                            past = int(row['past'])
                            if past not in args.pasts: continue
                            archives = []
                            if split == 1:
                                archives.append(repo/f'docs/experiments/SEQSCAN/raw/log/{model}_s{seq}_p{past}.txt')
                            elif args.order == 0 and seq == 128 and past == 3:
                                archives += [source.parent/f'{model}_k{split}_fp321_r{r}.txt' for r in range(50)]
                            if args.runtime_logs:
                                for r in range(args.runs):
                                    archives.append(args.runtime_logs/
                                        f'{model}_k{split}_order{args.order}_s{seq}_p{past}_r{r}.txt')
                            seen = 0
                            for archive in archives:
                                resources = records(archive,'E2E_RESOURCE')
                                schedules = records(archive,'E2E_SCHEDULE')
                                if len(resources) != len(schedules) or not resources:
                                    raise RuntimeError(f'incomplete archive: {archive}')
                                for res,schedule in zip(resources,schedules):
                                    if (res['grid'],res['block']) != (resource['grid'],resource['block']):
                                        raise RuntimeError('archive residency differs from symbolic projection')
                                    for field in ('task_refs','waits'):
                                        if int(row[field]) != int(schedule[field]):
                                            raise RuntimeError(f'A2 MISMATCH {archive} {field}: {row[field]} != {schedule[field]}')
                                    seen += 1
                            comparison = 'EXACT_RUNTIME_MATCH' if seen else 'NO_RUNTIME_CROSS_CELL_ARCHIVE'
                            writer.writerow(dict(model=model,**row,archive_processes=seen,comparison=comparison))
                            manifest['archive_comparisons'] += 2*seen
                            if not seen: manifest['unverified_cells'].append([model,seq,past,split])
                        stream.flush()
                        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                        print(f'{tag} exact counters checked; {manifest["commands"][-1]["seconds"]:.3f}s',flush=True)
        missing = len(manifest['unverified_cells'])
        (out/'status.txt').write_text(f'CPU PASS in requested scope; full A2 acceptance OPEN: '
                                    f'{missing} requested cross-cells lack runtime archives; '
                                    'other variants/ownership modes are not certified by this run\n')
    except BaseException:
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (out/'status.txt').write_text('FAILED or interrupted; full A2 acceptance OPEN\n')
        raise


if __name__ == '__main__':
    main()
