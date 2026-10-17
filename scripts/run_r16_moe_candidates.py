"""Strict full-model MoE MMA/WGMMA candidate driver with frozen source and shared checkpoint."""
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

def assert_implementation(identity, candidate, phase):
    if candidate['arm'] not in ('mma','h1') or candidate['mode'] != 'L2' or candidate['loop'] != 0:
        raise ValueError('MoE candidate requires explicit MMA/H1 and L2 host requests')
    expected = candidate['phase_sources'][phase]
    if set(expected) != {'root', 'head', 'compiler_sha256'} or any(identity['source'][k] != expected[k] for k in expected):
        raise ValueError('phase artifact source differs from explicit frozen binding')
    logical=identity.get('plan',{}).get('target_spec',{}).get('arch_tag')
    if logical != 'sm_90' or identity.get('arch') not in ('sm_90','sm_90a') or identity.get('nvcc_arch') != 'sm_90a':
        raise ValueError('candidate requires logical sm_90 and explicit sm_90a compilation')
    if int(identity.get('defines', {}).get('TILEMEGA_ARCH_ID', 800)) != 900:
        raise ValueError('seed TaskBody Caps still use ARCH_ID800; native binding was not applied')
    forced = int(identity.get('defines', {}).get('TILEMEGA_ARCH_PATH_SM80', 0))
    if bool(forced) != (candidate['arm'] == 'sm80'):
        raise ValueError('forced SM80 transport macro differs from candidate')
    plan = identity['plan']
    if plan.get('seed_arch') != 'native' or plan.get('seed_target_arch') != 'sm_90':
        raise ValueError('manifest does not certify explicit native seed binding')
    requested = 'wgmma' if phase == 'decode' and candidate['arm'] in ('h1', 'h1h4') else 'mma16'
    implementations = identity['implementations']['gemms']
    count = candidate['gemm_count']
    for rows in (implementations, plan['gemms']):
        if len(rows) != count or {row['index'] for row in rows} != set(range(count)):
            raise ValueError('full-model GEMM count or index coverage differs')
        if any(row['impl'] != ('mma16' if phase == 'decode' and candidate.get('lm_head_mma') and row['index'] == count-1 else requested) for row in rows):
            raise ValueError('requested GEMM implementation silently changed or fell back')
    arch_paths = 'sm80' if candidate['arm'] == 'sm80' else 'auto'
    if plan.get('arch_paths') != arch_paths or plan.get('pg') != 'off':
        raise ValueError('transport or executor layout differs from the candidate')
    layout=load(candidate['layout'])
    if 'shared_weight_layout_sha256' in plan and plan['shared_weight_layout_sha256'] != file_sha(candidate['layout']):
        raise ValueError('shared named weight recipe layout changed')
    # Dense compiler manifests omit the shared-layout SHA. Their actual
    # complete packed recipes and geometry still certify the same layout.
    expected={json.dumps(buffer['recipe']['source'],sort_keys=True):
              (buffer['recipe']['tile_n'],buffer['recipe']['tile_k']) for buffer in layout['buffers']}
    packed=[buffer['recipe'] for buffer in plan['buffers'] if buffer.get('recipe',{}).get('kind')=='tile_pages']
    actual={json.dumps(recipe['source'],sort_keys=True):(recipe['tile_n'],recipe['tile_k']) for recipe in packed}
    if len(packed)!=count or len(actual)!=count or any(expected.get(name)!=shape for name,shape in actual.items()):
        raise ValueError('actual full-model named packed weights differ from shared layout')
    if any((g['tile_n'],g['tile_k'])!=(64,64) for g in plan['gemms']):
        raise ValueError('actual GEMM geometry differs from the shared T64/K64 layout')
    if phase == 'decode':
        if (int(plan.get('dm_wgmma_inline',0)) != int(candidate['arm']=='h1') or plan.get('attention_noinline') is not True or
            bool(identity['implementations'].get('dm_wgmma_inline')) != (candidate['arm']=='h1') or
            int(identity['defines'].get('TILEMEGA_DM_NARROW_INLINE','0')) != int(candidate['arm']=='h1') or
            int(identity['defines'].get('TILEMEGA_DM_WGMMA_INLINE','0')) != int(candidate['arm']=='h1') or
            int(plan.get('moe_activation_bulk',0)) != int(candidate['arm']=='h1') or
            any(g['tile_m'] != 16 for g in plan['gemms'])):
            raise ValueError('formal H10 typed-inline flag or legal geometry missing')
        for coordinate, macro in (('activation_tma', 'TILEMEGA_ACTIVATION_TMA'),
                                  ('qkv_epilogue', 'TILEMEGA_QKV_EPILOGUE')):
            wanted = int(candidate.get(coordinate, 0))
            if int(plan.get(coordinate, 0)) != wanted or int(identity.get('defines', {}).get(macro, 0)) != wanted:
                raise ValueError(f'{coordinate} differs from the requested candidate')
        expected = 'wgmma' if candidate['arm'] == 'h1h4' else 'mma16'
        attention = identity['implementations']['attention']
        if attention['impl'] != expected or attention['page_layout'] != ('uniform' if expected == 'wgmma' else 'chunk16'):
            raise ValueError('requested decode attention silently changed or fell back')
        if int(plan.get('cluster_dim',1)) != 1 or int(identity['defines'].get('TILEMEGA_CLUSTER_COMBINE','0')):
            raise ValueError('unchanged MMA seed cannot request a cluster implementation')
        if requested == 'wgmma' and not all(identity['caps'].get(name) is True for name in ('wgmma', 'tma', 'bulk_copy')):
            raise ValueError('Hopper capability evidence is absent')
    if identity.get('trace') is not False:
        raise ValueError('instrumented binary cannot enter performance selection')

def check_pair(candidate):
    import shared_checkpoint_contract as shared
    if file_sha(shared.__file__) != candidate["shared_checkpoint_helper_sha256"]:
        raise ValueError("shared checkpoint helper changed")
    shared.validate(candidate,live=False)
    identities = {}
    for phase in ('prefill', 'decode'):
        identities[phase] = native_identity(candidate[phase])
        assert_implementation(identities[phase], candidate, phase)
    return identities

def execute(args):
    candidate = load(args.candidate)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if file_sha(__file__) != candidate['script_sha256']:
        raise ValueError('candidate driver changed after queue generation')
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
    if candidate['mode']!='L2' or candidate['loop']!=0:
        raise ValueError('entry-scoped MMA recovery only permits L2 host execution')
    entries, functions=resource_functions(warnings)
    resource={'evidence':'verified formal explicit H10 compiler resources; cold-path spills recorded; no default claim',
        'all_entries':list(entries.values()), 'all_functions':list(functions.values()),
        'ptxas_log_sha256':file_sha(candidate['decode']+'.ptxas.log'), 'binary_sha256':binary_sha,
        'acceptance_scope':'full-model correctness required; preexisting MMA spills recorded rather than relabeled zero'}
    if not entries:raise ValueError('compiler entry resource receipt absent')
    write(output/'selected-entry-resources.json',resource)
    gate=load(candidate['resource_gate_file'])
    if (file_sha(candidate['resource_gate_file']) != candidate['resource_gate_sha256'] or
        gate['binary_sha256'] != binary_sha or gate['C7510'] != 0 or
        (candidate['arm']=='h1' and (gate['audit']['hot_local_instructions'] != 0 or gate['audit']['counts'].get('HGMMA',0) <= 0 or gate['audit']['counts'].get('UBLKCP',0) <= 0)) or
        (candidate['arm']=='mma' and gate['audit']['counts'].get('HGMMA',0) != 0)):
        raise ValueError('formal H10 resource gate absent or belongs to another binary')
    write(output/'formal-h10-resource-gate.json',gate)
    if re.search(r'wgmma[^\n]*serializ', warnings, re.I):
        raise ValueError('ptxas serialized the requested WGMMA instructions')
    for phase, path in candidate['exports'].items():
        if file_sha(path) != candidate['export_sha256'][phase]:
            raise ValueError('export changed after queue preparation')
    if file_sha(Path(candidate['config'])/'config.json') != candidate['config_sha256']:
        raise ValueError('model config changed after queue preparation')
    import shared_checkpoint_contract as shared
    if file_sha(shared.__file__) != candidate['shared_checkpoint_helper_sha256']:
        raise ValueError('shared checkpoint helper changed')
    provenance=shared.validate(candidate,live=True)
    write(output/'shared-checkpoint-lease-validation.json',provenance)
    command = [candidate['python'], candidate['script'], 'native-run', '--model', candidate['shared_checkpoint']['model'], '--prefill', candidate['prefill'], '--decode', candidate['decode'],
        '--prompt-ids', candidate['prompt_ids'], '--batch', str(candidate['batch']), '--system-label', candidate['native_system_label'],
        '--max-tokens', '16' if args.stage=='smoke' else '1024', '--warmup', '1', '--repeats', '3',
        '--mode', candidate['mode'], '--decode-loop', str(candidate['loop']), '--round-index', str(args.round_index),
        '--out', str(output)]
    if args.stage in ('smoke', 'longcheck'):
        command += ['--check-executors', '--hf-check']
    if file_sha(Path(__file__).with_name('r16_argument_contract.py')) != candidate['argument_contract_sha256']:
        raise ValueError('native argument contract helper changed')
    parsed = validate_native_arguments(Path(candidate['snapshot_source'])/'python/tilemega/serving/r16_run.py', command[command.index('native-run')+1:])
    parsed.update(outer_system=candidate['system'], variant=candidate['variant'])
    write(output/'native-argument-contract.json', parsed)
    result = subprocess.run(command)
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
    from tilemega.serving import engine, r16_run, plan
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
    import shared_checkpoint_contract as shared
    shared.measurement(candidate,path)
    if load(path.parent.parent/'formal-h10-resource-gate.json') != load(candidate['resource_gate_file']):
        raise ValueError('measured H10 resource proof differs from frozen candidate')
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

def main():
    if sys.argv[1:2]==['native-run']:
        sys.argv.pop(1)
        return native_run()
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('execute');p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','pilot','final','longcheck'),required=True)
    p.add_argument('--round-index',type=int,default=0);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:execute(args)
    except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError) as error:
        write(args.out/'candidate-rejection.json',dict(evidence='verified rejection',error=str(error),candidate=str(args.candidate),stage=args.stage,fallback_substituted=False))
        raise SystemExit(3) from error
if __name__=='__main__':main()
