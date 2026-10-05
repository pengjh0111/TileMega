#!/usr/bin/env python3
"""Final CPU-only evidence review; never turn unavailable paths into passes."""
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import re
import select
import shlex
import signal
import statistics
import subprocess
import sys
import time

from bootstrap import HERE, ROOT, WORK, PY, sha, write
from pipeline import CELLS, read, trace_directories
from report import page_observations, table

OUT=HERE/'raw/acceptance_05'


def tsv(path):
    with path.open() as stream:
        yield from csv.DictReader(stream,delimiter='\t')


def archive(destination,sources):
    if destination.exists():raise RuntimeError('refuse to overwrite archive: '+str(destination))
    files=sorted({file for source in sources for file in
                  (source.rglob('*') if source.is_dir() else [source]) if file.is_file()})
    write(OUT/(destination.stem+'.files.json'),[dict(path=str(file),bytes=file.stat().st_size,
                                                   sha256=sha(file)) for file in files])
    destination.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['tar','-I','xz -T3 -1','-cf',str(destination),*map(str,sources)],check=True)
    record=dict(path=str(destination),sha256=sha(destination),bytes=destination.stat().st_size,
                files=len(files),time=time.time())
    write(OUT/(destination.stem+'.json'),record)
    return record


def snapshot():
    sources=[HERE/'raw'/row['name'] for row in read(HERE/'queue_trace_completion_r7.json')]
    sources += [HERE/'raw/acceptance_04/builds', HERE/'raw/acceptance_04/trace_builds.json',
                HERE/'raw/acceptance_04/trace_replacements.json']
    sources += [HERE/'results'/name for name in ('S6.tsv','S6_kinds.tsv','S7.tsv','S7_pages.tsv')]
    archive(OUT/'unreviewed_r7_evidence.tar.xz',sources)
    return 0


def inspect():
    state=read(WORK/'scheduler/state.json')
    if any(row['status'] in ('running','pending') for row in state.values()):
        raise RuntimeError('live experiment remains')
    nodes=read(HERE/'queue_trace_completion_r7.json')
    if any(state[row['name']]['status']!='done' or state[row['name']]['exit_code'] for row in nodes):
        raise RuntimeError('trace completion queue did not succeed')
    original={row['path']:row['sha256'] for row in read(HERE/'raw/acceptance_04/original_files.json')}
    preserved={}
    for name in ('measurements.tsv','S4.tsv','S5.tsv','S9.tsv','S10.tsv'):
        path=HERE/'results'/name
        preserved[name]=sha(path)
        if preserved[name]!=original[str(path)]:raise RuntimeError('performance/correctness changed: '+name)
    binaries={entry['path']:entry['sha256'] for rows in read(HERE/'catalog.json').values()
              for arm in rows for entry in arm.get('binaries',{}).values()}
    for path,expected in binaries.items():
        if sha(path)!=expected:raise RuntimeError('frozen binary changed: '+path)
    build_proof=[]
    originals={(row['cell'],row['label']):row for row in read(HERE/'builds_sm120_r3.json')}
    audit={(row['cell'],row['label']):row for row in read(HERE/'raw/E3_codegen_r5/audit.json')}
    for record in read(HERE/'raw/acceptance_04/trace_builds.json'):
        origin=originals[record['cell'],'P-R12bN-120']
        command=read(HERE/'raw/acceptance_04/builds'/record['cell']/'build.log.command.json')
        expected=shlex.split(Path(origin['so']+'.build_command.txt').read_text())
        expected[expected.index('-o')+1]=record['so']
        if command!=expected+['-DTILEMEGA_PAGE_TRACE=1','-DTILEMEGA_TRACE_STAGE=1','-DTILEMEGA_TRACE_STEP=1']:
            raise RuntimeError('trace flags changed beyond instrumentation')
        if record['exit_code'] or sha(record['so'])!=record['sha256'] or record['origin_sha256']!=origin['sha256']:
            raise RuntimeError('trace build identity mismatch')
        source=next(Path(arg) for arg in command if arg.endswith('.so.cu'))
        if sha(source)!=audit[record['cell'],'P-R12bN-120']['source_sha256']:
            raise RuntimeError('generated CUDA source changed')
        if sha(record['so']+'.plan.json')!=sha(origin['so']+'.plan.json'):
            raise RuntimeError('trace geometry metadata changed')
        build_proof.append(dict(cell=record['cell'],trace_sha256=record['sha256'],
                               source_sha256=sha(source),origin_sha256=record['origin_sha256']))
    traces=[]
    for stage in ('E5a','E5c'):
        for cell in ('llama_B1','qwen3_B1'):
            root=HERE/f'raw/{stage}_{cell}_r7'
            if read(root/'guard_result.json')['code']!=0:raise RuntimeError('trace outer guard is not clean')
            result=read(root/'trace_results.json')[0]
            folder=Path(result['path'])
            if result['exit_code'] or folder not in trace_directories(stage,cell):
                raise RuntimeError('trace replacement was not accepted')
            profiles=[]
            for file in sorted(folder.rglob('trace.json')):
                profile=read(file)
                expected_steps=1 if stage=='E5a' else 16
                if profile['mode']!=('L1' if stage=='E5a' else 'L2') or profile['loop'] or profile['steps']!=expected_steps:
                    raise RuntimeError('trace execution mismatch')
                iterations={int(row['iteration']) for row in tsv(file.parent/'step_trace.tsv')}
                if iterations!=set(range(16)):raise RuntimeError('trace does not contain sixteen launches/steps')
                if stage=='E5a':
                    stage_iterations={int(row['iteration']) for row in tsv(file.parent/'stage_trace.tsv')}
                    if stage_iterations!=set(range(16)):raise RuntimeError('TR-1 launch coverage incomplete')
                profiles.append(profile)
            if sorted(row['past'] for row in profiles)!=([64,575,1000] if stage=='E5a' else [575]):
                raise RuntimeError('trace past coverage mismatch')
            active=Counter(int(row['step']) for row in tsv(folder/'page_trace.tsv')
                           if int(row['kernel_end_ns'])>int(row['kernel_begin_ns'])>0)
            expected=({936:170} if stage=='E5a' else {step:170 for step in range(511,527)})
            if dict(active)!=expected:raise RuntimeError('page-step/worker coverage mismatch')
            traces.append(dict(stage=stage,cell=cell,profiles=profiles,page_active_workers=dict(active),
                  page_trace_sha256=sha(folder/'page_trace.tsv'),outer_guard=read(root/'guard_result.json'),
                  limitation='TR-1 page export retains past1000 only and accumulates 16 launches'
                             if stage=='E5a' else 'one launch per each of sixteen steps'))
    write(OUT/'review.json',dict(time=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          r7_nodes={row['name']:state[row['name']] for row in nodes},preserved_tables=preserved,
          frozen_binary_count=len(binaries),trace_build_proof=build_proof,traces=traces,
          no_new_gpu_measurements=True,no_solver_changes=True,no_push=True,
          verdict='accept executed evidence with documented unavailable paths; not full protocol PASS'))
    print('reviewed 6 terminal nodes, 27 frozen binaries, 2 trace rebuilds and 4 clean traces')
    return 0


def pages():
    ceiling=read(HERE/'raw/E1_TL2_r3/processes/dram_ceiling.json')
    rows=[]
    for cell in ('llama_B1','qwen3_B1'):
        floor=read(HERE/f'raw/E3_catalog_r5/floors/{cell}/decode/floor.json')
        for stage in ('E5a','E5c'):
            for folder in trace_directories(stage,cell):
                for file in folder.rglob('page_trace.tsv'):
                    rows.extend(dict(row,cell=cell) for row in page_observations(file,floor['points'],
                                {'native':ceiling['calibration_median_gbps'],'mb1a_max':ceiling['maximum_gbps']}))
    table('S7_pages.tsv',rows)
    print('corrected page observations:',len(rows),'(raw counters unchanged)')
    return 0


def semantic(row):
    if row['kind']!='kGemm':return row['kind']
    return next((label for token,label in (('qkv','qkv'),('o_proj','o'),('gate_up','gate_up'),
                ('down_proj','down'),('lm_head','lm_head')) if token in row['name']),row['kind'])


def tables():
    header=ROOT/'third_party/cutlass/include/cutlass/version.h'
    text=header.read_text()
    version='.'.join(re.search(r'#define CUTLASS_'+field+r' (\d+)',text)[1]
                     for field in ('MAJOR','MINOR','PATCH'))
    table('S1_vendor.tsv',[dict(distribution='CUTLASS/CuTe local headers',version=version,
           header=str(header),sha256=sha(header),git_commit=None,
           qualification='verified local header macros; vendor directory is not an independent Git checkout')])
    ceilings={row['architecture']:float(row['calibration_median_gbps']) for row in tsv(HERE/'results/S2_ceiling.tsv')}
    stages=defaultdict(lambda:defaultdict(list))
    for row in tsv(HERE/'results/S6_kinds.tsv'):
        if '/E5a_' not in row['source']:continue
        arm=Path(row['source']).parent.name
        stages[row['cell'],arm,int(row['past']),row['kind']]['sm120'].append(float(row['excess_ns_native']))
    old=defaultdict(float)
    for row in tsv(HERE/'results/S6_sm89_reference.tsv'):
        arm=Path(row['source']).parent.name
        if 'L1-loop0' not in arm:continue
        arm='PR_L1' if arm.startswith('P-R12bN-') else 'B0'
        key=(row['cell'],arm,int(row['past']),semantic(row),int(row['iteration']))
        old[key]+=float(row['duration_ns'])-int(row['bytes'])/ceilings['sm89']
    for key,value in old.items():stages[key[:-1]]['sm89'].append(value)
    table('S6_cross_arch.tsv',[dict(cell=key[0],arm=key[1],past=key[2],kind=key[3],
        **{arch+'_excess_us':statistics.median(values.get(arch,[]))/1000 if values.get(arch) else None
           for arch in ('sm120','sm89')},
        qualification='diagnostic instrumentation only; each architecture uses its own TL-2 ceiling; missing is n/a')
        for key,values in sorted(stages.items())])
    steps=defaultdict(lambda:defaultdict(list))
    for row in tsv(HERE/'results/S7.tsv'):
        if '/E5c_' in row['source']:
            steps[row['cell'],Path(row['source']).parent.name]['sm120'].append(row)
    for row in tsv(HERE/'results/S7_sm89_reference.tsv'):
        if '/B3trace_' not in row['source']:continue
        cell=next(cell for cell in CELLS if '_'+cell+'/' in row['source'])
        label=Path(row['source']).parent.name
        arm=('PR_L2l' if 'loop1' in label else 'PR_L2') if label.startswith('P-') else ('B0l' if 'loop1' in label else 'B0-noev')
        steps[cell,arm]['sm89'].append(row)
    output=[]
    for (cell,arm),values in sorted(steps.items()):
        result=dict(cell=cell,arm=arm,qualification='TR-3 diagnostic medians; missing lag stamps are not proof of zero wait')
        for arch in ('sm120','sm89'):
            for field in ('boundary_gap_ns','token_lag_mean_ns','kv_lag_mean_ns','last_barrier_wait_mean_ns'):
                numbers=[float(row[field])/1000 for row in values.get(arch,[]) if row[field]]
                result[arch+'_'+field.removesuffix('_ns')+'_us']=statistics.median(numbers) if numbers else None
        output.append(result)
    table('S7_cross_arch.tsv',output)
    return 0


def binaries():
    entries={entry['path']:entry['sha256'] for rows in read(HERE/'catalog.json').values()
             for arm in rows for entry in arm.get('binaries',{}).values()}
    entries.update({row['so']:row['sha256'] for row in read(HERE/'raw/acceptance_04/trace_builds.json')})
    sources=[]
    for path,expected in entries.items():
        if sha(path)!=expected:raise RuntimeError('binary changed before archive')
        sources.append(Path(path))
        sources += [Path(path+suffix) for suffix in ('.plan.json','.serving.json','.cu','.classes.tsv','.build_command.txt')
                    if Path(path+suffix).exists()]
    destination=Path('/root/shared-nvme/junhuipeng/TileMega_R13_SM120/artifacts/r13_sm120_frozen_binaries.tar.xz')
    archive(destination,sources)
    print('persisted',len(entries),'binary artifacts in shared storage')
    return 0


def stop():
    state=read(WORK/'scheduler/state.json')
    for file in (WORK/'queue').glob('queue_*.json'):
        if any(state.get(row['name'],{}).get('status') not in ('done','failed','skipped','not_run')
               for row in read(file)):
            raise RuntimeError('registered work is not terminal')
    pid=4488
    descriptor=os.pidfd_open(pid)
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if str(HERE.with_name('SERVING_R13')/'scheduler.py') not in command or str(WORK/'queue') not in command:
            raise RuntimeError('scheduler PID identity mismatch')
        children=subprocess.run(['ps','--ppid',str(pid),'-o','pid='],capture_output=True,text=True).stdout.strip()
        if children:raise RuntimeError('scheduler still has children')
        signal.pidfd_send_signal(descriptor,signal.SIGTERM)
        exited=bool(select.select([descriptor],[],[],5)[0])
        if not exited:raise RuntimeError('scheduler did not exit after SIGTERM')
        write(OUT/'scheduler_shutdown.json',dict(time=time.time(),pid=pid,command=command,
              all_registered_work_terminal=True,children_before=[],signal='SIGTERM',exited=True,
              no_gpu_reset=True,no_state_cleared=True))
    finally:os.close(descriptor)
    return 0


if __name__=='__main__':raise SystemExit(globals()[sys.argv[1]]())
