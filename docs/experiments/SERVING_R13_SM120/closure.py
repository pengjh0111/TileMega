#!/usr/bin/env python3
"""Accept terminal evidence and repair only the prescribed PR trace collection."""
import argparse
from collections import Counter, defaultdict
import concurrent.futures
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from bootstrap import HERE, ROOT, FRAME, WORK, PY, sha, write, run
from pipeline import CELLS, MATRICES, GROUPS, read, runnable, round_path

FOLDER = HERE / 'raw/acceptance_04'
REPAIRS = ('llama_B1', 'qwen3_B1')


def coverage(rows, available):
    counts = Counter((row['matrix'], row['cell'], row['arm']) for row in rows)
    expected = {(matrix, cell, arm) for matrix, arms in MATRICES.items()
                for cell in CELLS for arm in arms
                if not (cell == 'qwen3_B16' and arm == 'B0h')}
    eligible = expected & set(available)
    errors = [dict(key=key, records=counts[key], expected=3) for key in sorted(eligible)
              if counts[key] != 3]
    errors += [dict(key=key, records=value, error='unregistered/ineligible measurement')
               for key, value in counts.items() if key not in eligible]
    return dict(expected_records=len(expected)*3, eligible_records=len(eligible)*3,
                measured_records=sum(counts.values()), unavailable_records=len(expected-eligible)*3,
                all_eligible_collected=not errors, errors=errors)


def snapshot():
    destination = FOLDER / 'original_e4_e5_e6.tar.xz'
    if destination.exists():
        raise RuntimeError('refuse to overwrite the original evidence archive')
    sources = sorted(folder for folder in (HERE/'raw').iterdir()
                     if folder.is_dir() and folder.name.startswith(('E4', 'E5', 'E6')))
    sources += [HERE/'results', HERE/'summary.generated.md', HERE/'summary.md',
                HERE/'builds_sm120_r3.json', HERE/'catalog.json',
                WORK/'scheduler/state.json', WORK/'scheduler/progress.tsv']
    files = sorted({file for source in sources for file in
                    (source.rglob('*') if source.is_dir() else [source]) if file.is_file()})
    write(FOLDER/'original_files.json', [dict(path=str(file), bytes=file.stat().st_size,
                                            sha256=sha(file)) for file in files])
    # Absolute paths are stripped by tar; no active task or weight file is included.
    subprocess.run(['tar', '-I', 'xz -T3 -1', '-cf', str(destination),
                    *map(str, sources)], cwd=ROOT, check=True)
    write(FOLDER/'archive.json', dict(path=str(destination), bytes=destination.stat().st_size,
          sha256=sha(destination), source_files=len(files), time=time.time(),
          original_files_unchanged=True, excludes=['weights', 'temporary caches']))
    return 0


def audit():
    from report import measurements, table
    state = read(WORK/'scheduler/state.json', {})
    original = read(HERE/'queue_complete_r5.json', [])
    write(FOLDER/'terminal_state.json', dict(time=time.time(), original_queue_nodes=len(original),
          original_queue_statuses=dict(Counter(state.get(row['name'],{}).get('status','absent')
                                               for row in original)),
          active={key:value for key,value in state.items() if value['status'] in ('running','pending')},
          state=state))
    rows, _ = measurements()
    available = set()
    domain = []
    for matrix, arms in MATRICES.items():
        for cell in CELLS:
            for label in arms:
                if cell=='qwen3_B16' and label=='B0h':continue
                arm = runnable(cell,label)
                if arm['available']:available.add((matrix,cell,label))
                domain.append(dict(matrix=matrix,cell=cell,arm=label,available=arm['available'],
                              reason=arm.get('unavailable_reason')))
    result = coverage(rows, available)
    table('acceptance_coverage.tsv', domain)
    protocols = {}
    all_pids = []
    for group in GROUPS:
        folder = HERE/f'raw/E2c_{group}_r5'
        records = read(folder/'processes/processes.json', [])
        status = read(folder/'status.json', {})
        pids = [record['pid'] for record in records if record.get('execution_complete')]
        all_pids += pids
        protocols[group] = dict(status=status, independently_counted=len(records),
              unique_pids=len(set(pids)), all_exit_zero=all(row.get('exit_code')==0 for row in records),
              actual_pids=pids, unavailable_not_a_pass=not status.get('complete',False))
        if status.get('complete') and (len(pids)!=50 or len(set(pids))!=50 or
                                       not all(row.get('passed') for row in records)):
            result['errors'].append(dict(group=group,error='fresh-process evidence disagrees with status'))
    canonical = []
    for matrix in MATRICES:
        for cell in CELLS:
            for number in range(3):
                path=round_path(matrix,cell,number)
                if path.exists():canonical.append(dict(matrix=matrix,cell=cell,round=number,
                                                       path=str(path),sha256=sha(path)))
    checks = [row for cell in CELLS for row in read(HERE/f'raw/E4f_{cell}_r5/checks.json',[])]
    c2 = [row for row in checks if row.get('check')=='C-2' and row.get('pass_') is not None]
    nonself = [row for row in c2 if row['label']!=row['reference']]
    cluster = [row for row in read(HERE/'raw/E2a_MB-1d_r3/loadbench.json')['points']
               if row['suite']=='MB-1d-cluster']
    result.update(time=time.time(), canonical_rounds=canonical, protocols=protocols,
          global_protocol_unique_pids=len(set(all_pids)), global_protocol_count=len(all_pids),
          C2=dict(recorded=len(c2),passed=sum(bool(row['pass_']) for row in c2),
                  nonself_comparisons=len(nonself),nonself_passed=sum(bool(row['pass_']) for row in nonself)),
          cluster=dict(requested=len(cluster),executed=sum(row.get('status')!='unsupported' for row in cluster),
                       unsupported=sum(row.get('status')=='unsupported' for row in cluster),
                       reason='170 foreground CTAs / 85 with background do not divide by most cluster sizes',
                       fresh_process_sync_claim=False),
          final_full_protocol_pass=False, human_final_report_pending=True,
          trace_correction_pending=not (FOLDER/'trace_replacements.json').exists(),
          limitations=['B16 paged and all R13F plans unavailable after negative traffic rejection',
                       'paged/FX21 fresh-process groups unavailable',
                       'B0h Llama B1 HF check fails unchanged reference thresholds',
                       'five CTest items remain unresolved',
                       'cross-host model SHA not supplied; vLLM 0.29.0 vs 0.30.0',
                       'trace overhead has no matched uninstrumented fixed-past control; diagnostic only',
                       'MB-1e negative cold-stream residual is not physical boundary latency'])
    result['all_eligible_collected'] = not result['errors']
    write(FOLDER/'acceptance.json',result)
    print(json.dumps({key:result[key] for key in ('expected_records','eligible_records',
          'measured_records','unavailable_records','all_eligible_collected','errors','C2','cluster')},ensure_ascii=False))
    return int(bool(result['errors']))


def trace_command(source, destination):
    command=shlex.split(source)
    command[command.index('-o')+1]=str(destination)
    return command+['-DTILEMEGA_PAGE_TRACE=1','-DTILEMEGA_TRACE_STAGE=1','-DTILEMEGA_TRACE_STEP=1']


def build_one(cell):
    original = next(row for row in read(HERE/'builds_sm120_r3.json')
                    if row['cell']==cell and row['label']=='P-R12bN-120')
    if original['exit_code'] or sha(original['so'])!=original['sha256']:
        raise RuntimeError('PR origin unavailable or changed: '+cell)
    folder = Path(original['so']).parent.parent/'P-R12bN-120-trace-r7'
    destination = folder/'plan.so'
    if destination.exists():raise RuntimeError('refuse to overwrite trace output')
    folder.mkdir(parents=True,exist_ok=True)
    command = trace_command(Path(original['so']+'.build_command.txt').read_text(),destination)
    code=run(command,FOLDER/'builds'/cell/'build.log')
    record=dict(original,label='P-R12bN-120-trace',so=str(destination),exit_code=code,
                page_trace_enabled=True,origin_sha256=original['sha256'],trace_only=True)
    if not code:
        record['sha256']=sha(destination)
        for suffix in ('.plan.json','.cu','.classes.tsv'):
            source=Path(original['so']+suffix)
            if source.exists():
                subprocess.run(['cp',str(source),str(destination)+suffix],check=True)
    write(FOLDER/'builds'/cell/'record.json',record)
    return record


def build():
    # Two prescribed trace rebuilds, no search or new geometry; performance binaries stay frozen.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(build_one,REPAIRS))
    write(FOLDER/'trace_builds.json',records)
    return int(any(row['exit_code'] for row in records))


def retrace(stage,cell):
    from pipeline import trace
    label='PR_L1' if stage=='E5a' else 'PR_L2'
    code=trace(stage,cell,revision='r7',labels=[label],
               build_overrides=read(FOLDER/'trace_builds.json',[]))
    folder=HERE/f'raw/{stage}_{cell}_r7'
    results=read(folder/'trace_results.json',[])
    if code or not results or any(row.get('exit_code')!=0 for row in results):return code or 1
    replacements=read(FOLDER/'trace_replacements.json',{})
    replacements[f'{stage}:{cell}']=dict(results=str(folder/'trace_results.json'),
        guard=str(folder/'guard_result.json'),reason='PR trace flags omitted PAGE_TRACE; original trace preserved')
    write(FOLDER/'trace_replacements.json',replacements)
    return 0


def collect():
    from report import collect as report
    code=report(HERE/'raw/E6_collect_r7')
    return code or audit()


def register():
    state=read(WORK/'scheduler/state.json')
    if any(row['status'] in ('running','pending') for row in state.values()):
        raise RuntimeError('original queue is not terminal')
    if not read(FOLDER/'archive.json',{}).get('sha256'):
        raise RuntimeError('original final collection has not been archived')
    environment=read(HERE/'launch_env_r3.json')
    nodes=[]
    def add(name,action,gpu=False,arguments=(),after=(),after_any=()):
        nodes.append(dict(name=name,command=[PY,str(HERE/'closure.py'),action,*arguments],
          cwd=str(ROOT),env=environment,gpu=gpu,after=list(after),after_any=list(after_any),
          priority=65,timeout_s=7200,needs_free_mib=12288,out=str(HERE/'raw'/name)))
    add('E3_PR_trace_r7','build')
    traces=[]
    for stage in ('E5a','E5c'):
        for cell in REPAIRS:
            name=f'{stage}_{cell}_r7';traces.append(name)
            add(name,'trace',True,['--stage',stage,'--cell',cell],after=['E3_PR_trace_r7'])
    add('E6_collect_r7','collect',after_any=['E3_PR_trace_r7',*traces])
    destination=WORK/'queue/queue_trace_completion_r7.json'
    if destination.exists():raise RuntimeError('trace correction already published')
    write(HERE/'queue_trace_completion_r7.json',nodes)
    temporary=destination.with_suffix('.pending');write(temporary,nodes);temporary.replace(destination)
    write(FOLDER/'publication.json',dict(time=time.time(),nodes=[row['name'] for row in nodes],
          unchanged_measurements=195,only_trace_recollection=True,no_solver_changes=True,no_push=True))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('snapshot','audit','build','trace','collect','register'))
    parser.add_argument('--stage',choices=('E5a','E5c'))
    parser.add_argument('--cell',choices=REPAIRS)
    args=parser.parse_args()
    raise SystemExit(retrace(args.stage,args.cell) if args.action=='trace' else globals()[args.action]())
