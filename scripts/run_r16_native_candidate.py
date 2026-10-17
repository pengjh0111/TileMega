#!/usr/bin/env python3
"""Run an immutable native candidate using an independently frozen Python runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from r16_measurement_validation import completed_guard, file_sha, native_identity, validate_llm
from r16_argument_contract import validate_native_arguments

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def execute(args):
    candidate = load(args.candidate)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if file_sha(__file__) != candidate['script_sha256']:
        raise ValueError('candidate driver changed after queue generation')
    helper=Path(__file__).with_name('r16_runtime_contract.py')
    if file_sha(helper)!=candidate['runtime_contract_sha256']:
        raise ValueError('runtime contract helper changed')
    import r16_runtime_contract as runtime
    runtime.validate_source(candidate['runtime_source'])
    identities = check_pair(candidate)
    metadata = dict(system=candidate['system'], native_system_label=candidate['native_system_label'], variant=candidate['variant'], stage=args.stage, candidate=candidate['id'] if 'id' in candidate else candidate['arm'],
        artifact_ids={phase:value['artifact_id'] for phase,value in identities.items()}, seed=candidate['seed'],
        config_sha256=candidate['config_sha256'], export_sha256=candidate['export_sha256'],
        shared_weight_layout_sha256=file_sha(candidate['layout']))
    write(output/'candidate-evidence.json', metadata)
    cache = Path(candidate['decode']+'.r16-sass-counts.json')
    cached = load(cache) if cache.exists() else {}
    binary_sha = file_sha(candidate['decode'])
    if cached.get('binary_sha256') != binary_sha:
        command = ['/usr/local/cuda/bin/cuobjdump', '--dump-sass', candidate['decode']]
        sass = subprocess.run(command, capture_output=True, text=True, check=True).stdout
        cached = dict(binary_sha256=binary_sha, command=command,
            sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),
            instructions={name:len(re.findall(r'\b'+name+r'\b', sass)) for name in ('HGMMA','UTMALDG','UBLKCP')})
        write(cache,cached)
    counts=cached['instructions']
    write(output/'sass-evidence.json', cached)
    if candidate['arm'] in ('h1', 'h1h4') and counts['HGMMA'] == 0:
        raise ValueError('WGMMA candidate binary contains no HGMMA instruction')
    warnings = Path(candidate['decode']+'.ptxas.log').read_text()
    from r16_kernel_resources import resource_functions
    helper=Path(__file__).with_name('r16_kernel_resources.py')
    if file_sha(helper)!=candidate['kernel_resource_helper_sha256']:
        raise ValueError('selected-entry resource helper changed')
    if candidate['mode'] not in ('L1','L2') or candidate['loop'] not in (0,1):
        raise ValueError('unsupported explicit native executor')
    entries, functions=resource_functions(warnings)
    resource={'evidence':'verified compiler resource records for unchanged SM80 compute seed; no Hopper/default claim',
        'all_entries':list(entries.values()), 'all_functions':list(functions.values()),
        'ptxas_log_sha256':file_sha(candidate['decode']+'.ptxas.log'), 'binary_sha256':binary_sha,
        'acceptance_scope':'full-model correctness required; preexisting MMA spills recorded rather than relabeled zero'}
    if not entries:raise ValueError('compiler entry resource receipt absent')
    write(output/'selected-entry-resources.json',resource)
    if re.search(r'wgmma[^\n]*serializ', warnings, re.I):
        raise ValueError('ptxas serialized the requested WGMMA instructions')
    for phase, path in candidate['exports'].items():
        if file_sha(path) != candidate['export_sha256'][phase]:
            raise ValueError('export changed after queue preparation')
    if file_sha(Path(candidate['config'])/'config.json') != candidate['config_sha256']:
        raise ValueError('model config changed after queue preparation')
    command = [candidate['python'], '-m', 'tilemega.serving.random_checkpoint', '--config', candidate['config'],
        '--out', str(output.parent/'checkpoint'), '--seed', str(candidate['seed']), '--max-gib', '64', '--',
        candidate['python'], candidate['script'], 'native-run', '--prefill', candidate['prefill'], '--decode', candidate['decode'],
        '--prompt-ids', candidate['prompt_ids'], '--batch', str(candidate['batch']), '--system-label', candidate['native_system_label'],
        '--max-tokens', '16' if args.stage=='smoke' else '1024', '--warmup', '1', '--repeats', '3',
        '--mode', candidate['mode'], '--decode-loop', str(candidate['loop']), '--round-index', str(args.round_index),
        '--out', str(output)]
    if args.stage in ('smoke', 'longcheck'):
        command += ['--check-executors', '--hf-check']
    if file_sha(Path(__file__).with_name('r16_argument_contract.py')) != candidate['argument_contract_sha256']:
        raise ValueError('native argument contract helper changed')
    parsed = validate_native_arguments(Path(candidate['runtime_source']['root'])/'python/tilemega/serving/r16_run.py', command[command.index('native-run')+1:])
    parsed.update(outer_system=candidate['system'], variant=candidate['variant'])
    write(output/'native-argument-contract.json', parsed)
    environment=dict(os.environ, TILEMEGA_R16_RUNTIME_SOURCE=json.dumps(candidate['runtime_source']), PYTHONPATH=str(Path(candidate['runtime_source']['root'])/'python'), PYTHONNOUSERSITE='1')
    result = subprocess.run(command, env=environment)
    if result.returncode:
        write(output/'candidate-rejection.json', dict(metadata, exit_code=result.returncode,
            reason='official compile/runtime/numerical path rejected; no fallback was substituted'))
    raise SystemExit(result.returncode)


def execution_row(engine, result, count):
    expected=bool(engine.decode_loop and count>1)
    if bool(result.decode_loop_used)!=expected:
        raise ValueError('requested device-loop execution silently fell back to host launches')
    return {'tokens':count,'executor':'L1' if engine.decode_mode==1 else 'L2',
            'requested_device_loop':expected,'device_loop_used':bool(result.decode_loop_used)}


def native_run():
    import r16_runtime_contract as runtime
    source=json.loads(os.environ['TILEMEGA_R16_RUNTIME_SOURCE'])
    runtime.validate_source(source)
    from tilemega.serving import engine, r16_run, plan
    runtime.attest(source)
    from tilemega.serving.baseline_common import guarded_main
    original=engine.ServingEngine.generate
    original_launch=plan.Plan.launch
    original_steps=plan.Plan.launch_steps
    rows=[]
    active={}
    def launch(current,step,mode,stream):
        if active.get('engine') is not None and current is active['engine'].decode:
            active['host_calls']+=1
            active['host_modes'].add(mode)
        return original_launch(current,step,mode,stream)
    def steps(current,first,count,mode,stream):
        if active.get('engine') is not None and current is active['engine'].decode:
            active['loop_calls']+=1
        return original_steps(current,first,count,mode,stream)
    def checked(current,prompt_ids,new_tokens=None):
        count=new_tokens or current.max_new_tokens
        active.update(engine=current,host_calls=0,loop_calls=0,host_modes=set())
        result=original(current,prompt_ids,new_tokens)
        row=execution_row(current,result,count)
        if not row['device_loop_used']:
            if active['host_calls']!=count-1 or active['loop_calls'] or (count>1 and active['host_modes']!={current.decode_mode}):
                raise ValueError('actual Plan.launch calls disagree with selected host executor')
        elif active['host_calls'] or active['loop_calls']!=1:
            raise ValueError('actual Plan.launch_steps calls disagree with selected device loop')
        row.update(actual_decode_host_calls=active['host_calls'],actual_decode_loop_calls=active['loop_calls'],
                   actual_decode_host_modes=sorted(active['host_modes']),
                   selected_host_kernel=('tilemega_l2_kernel' if current.decode_mode==2 else 'tilemega_l1_kernel') if not row['device_loop_used'] else None)
        rows.append(row)
        active.clear()
        return result
    engine.ServingEngine.generate=checked
    plan.Plan.launch=launch
    plan.Plan.launch_steps=steps
    output=Path(sys.argv[sys.argv.index('--out')+1])
    try:guarded_main(r16_run.main)
    finally:
        write(output/'runtime-execution.json',runtime.attest(source))
        engine.ServingEngine.generate=original
        plan.Plan.launch=original_launch
        plan.Plan.launch_steps=original_steps
        write(output/'actual-execution.json',{'evidence':'verified runtime Generation and counted actual Plan.launch/launch_steps API calls; source-bound tm_plan_launch dispatch',
            'driver_sha256':file_sha(__file__),'calls':rows})


def measurement(root, state, name, candidate):
    record = state[name]
    directory = Path(record['out'])
    if not directory.is_absolute():
        directory = root/directory
    guard = completed_guard(directory, record)
    path = directory/'measurements'/f"B{candidate['batch']}"/'measurements.json'
    value, _, ownership = validate_llm(path, guard, root)
    actual=load(directory/'measurements/actual-execution.json')
    calls=actual['calls']
    if actual['driver_sha256']!=candidate['script_sha256'] or len(calls)!=8:
        raise ValueError('actual executor evidence is absent or belongs to another driver')
    for count in (1,1024):
        group=[row for row in calls if row['tokens']==count]
        if len(group)!=4 or any(row['executor']!=candidate['mode'] or
            row['device_loop_used']!=bool(candidate['loop'] and count>1) for row in group):
            raise ValueError('actual runtime execution differs from selected executor/device loop')
    expected = check_pair(candidate)
    for phase in ('prefill', 'decode'):
        if value['configuration']['identities'][phase]['artifact_id'] != expected[phase]['artifact_id']:
            raise ValueError('measurement belongs to another candidate binary')
    config = value['configuration']
    if (value['model_config_sha256'] != candidate['config_sha256'] or value['gpu_index'] != candidate['gpu'] or
            config['mode'] != candidate['mode'] or config['decode_loop'] != candidate['loop'] or
            value['prompt_ids_sha256'] != file_sha(candidate['prompt_ids'])):
        raise ValueError('measurement workload/executor/card differs from candidate')
    checkpoint = load(config['checkpoint_manifest'])
    if checkpoint['seed'] != candidate['seed']:
        raise ValueError('measurement random checkpoint seed differs')
    return dict(candidate=candidate['id'], e2e_seconds=value['e2e_seconds'],
        tpot_seconds=value['tpot_seconds'], ttft_seconds=value['ttft_seconds'],
        tokens_per_second=value['output_tokens_per_second'], pid=value['pid'],
        measurement=str(path), checkpoint_sha256=checkpoint['checkpoint_sha256'],
        versions=value['versions'], artifact_ids={p:x['artifact_id'] for p,x in expected.items()},
        acceptance=ownership['acceptance'])




def check_pair(candidate):
    identities={}
    for phase in ('prefill','decode'):
        binary=Path(candidate[phase])
        if file_sha(str(binary)+'.identity.json')!=candidate['identity_sha256'][phase]:
            raise ValueError('candidate phase identity changed')
        identity=native_identity(binary)
        if any(identity['source'].get(k)!=v for k,v in candidate['phase_sources'][phase].items()):
            raise ValueError('compiled artifact source differs from frozen phase binding')
        if identity.get('trace') is not False or identity['nvcc_arch']!='sm_90a':
            raise ValueError('timing candidate requires uninstrumented sm_90a binary')
        if len(identity['plan']['gemms'])!=candidate['gemm_count']:
            raise ValueError('candidate is not the declared full model')
        identities[phase]=identity
    return identities


def main():
    if len(sys.argv)>1 and sys.argv[1]=='native-run':
        sys.argv.pop(1)
        native_run()
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['execute'])
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--stage',choices=['smoke','longcheck','pilot','final'],required=True)
    parser.add_argument('--round-index',type=int,default=0)
    parser.add_argument('--out',type=Path,required=True)
    execute(parser.parse_args())


if __name__=='__main__':
    main()
