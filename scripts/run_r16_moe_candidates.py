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
    if candidate['arm'] not in ('mma','h1') or (candidate['mode'],int(candidate['loop'])) not in {('L2',0),('L1',1)}:
        raise ValueError('MoE candidate requires explicit MMA/H1 and L2 host or L1 device loop')
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
    tile_n = int(candidate.get('packing_tile_n', 64))
    if tile_n not in (64, 128) or (candidate['arm']=='h1' and tile_n!=64):
        raise ValueError('unsupported MoE family packing geometry')
    if any((g['tile_n'],g['tile_k'])!=(tile_n,64) for g in plan['gemms']):
        raise ValueError('actual GEMM geometry differs from the shared T64/K64 layout')
    if phase == 'decode':
        if (int(plan.get('dm_wgmma_inline',0)) != int(candidate['arm']=='h1') or bool(plan.get('attention_noinline')) != bool(candidate.get('attention_noinline',1)) or
            bool(identity['implementations'].get('dm_wgmma_inline')) != (candidate['arm']=='h1') or
            int(identity['defines'].get('TILEMEGA_DM_NARROW_INLINE','0')) != int(candidate['arm']=='h1') or
            int(identity['defines'].get('TILEMEGA_DM_WGMMA_INLINE','0')) != int(candidate['arm']=='h1') or
            int(plan.get('moe_activation_bulk',0)) != int(candidate['arm']=='h1') or
            any(g['tile_m'] != 16 for g in plan['gemms'])):
            raise ValueError('formal H10 typed-inline flag or legal geometry missing')
        for coordinate, macro in (('moe_weight_bulk', 'TILEMEGA_MOE_WEIGHT_BULK'),
                                  ('activation_tma', 'TILEMEGA_ACTIVATION_TMA'),
                                  ('qkv_epilogue', 'TILEMEGA_QKV_EPILOGUE')):
            wanted = int(candidate.get(coordinate, 0))
            if int(plan.get(coordinate, 0)) != wanted or int(identity.get('defines', {}).get(macro, 0)) != wanted:
                raise ValueError(f'{coordinate} differs from the requested candidate')
        if candidate.get('moe_weight_bulk') and candidate['arm']!='h1':
            raise ValueError('expert weight bulk requires the explicit H10 WGMMA family')
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

def runtime_contract(candidate):
    if 'runtime_source' not in candidate:
        return None
    path = Path(__file__).with_name('r16_runtime_contract.py')
    if file_sha(path) != candidate['runtime_contract_sha256']:
        raise ValueError('runtime contract helper changed')
    import importlib.util
    spec=importlib.util.spec_from_file_location('moe_frozen_runtime_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.validate_source(candidate['runtime_source'])
    return module


def validate_loop_reference(candidate, identities):
    if not candidate['loop']:
        return
    binding=candidate.get('numeric_reference')
    if not binding or file_sha(binding['path'])!=binding['sha256']:
        raise ValueError('device-loop timing requires a frozen numerical reference')
    proof=load(binding['path'])
    for row in proof['files']:
        if file_sha(row['path'])!=row['sha256']:
            raise ValueError('device-loop numerical reference changed')
    host=load(proof['host_candidate'])
    if file_sha(proof['host_candidate'])!=proof['host_candidate_sha256']:
        raise ValueError('reference candidate changed')
    for key in ('prefill','decode','arm','batch','seed','config_sha256','shared_checkpoint','phase_sources'):
        if host[key]!=candidate[key]:
            raise ValueError('loop reference belongs to another body or checkpoint: '+key)
    state=load(proof['state_file']);record=state[proof['longcheck_job']]
    if record['status']!='done' or record.get('exit_code')!=0:
        raise ValueError('reference execution is no longer accepted')
    path=Path(proof['measurement']);value=load(path)
    if (path.parent/'invalid_measurements.json').exists():
        raise ValueError('reference measurements were invalidated')
    artifacts={p:i['artifact_id'] for p,i in identities.items()}
    if {p:i['artifact_id'] for p,i in value['configuration']['identities'].items()}!=artifacts:
        raise ValueError('reference binary differs from loop candidate')
    source=candidate['runtime_source']
    if any(value['configuration'].get(k)!=source[k] for k in ('runner_sha256','hf_checker_sha256')):
        raise ValueError('reference Python entry differs from loop runtime')
    directory=path.parent.parent
    generation=next(r for r in value['generation_runs'] if r['N']==1024 and r['run']==3)
    required={Path(proof['host_candidate']).resolve(),path.resolve()} | {
        (directory/name).resolve() for name in ('c1.json','c2.json','actual-execution.json','tokens_L1_loop1.json')}
    required.add((path.parent/generation['tokens_file']).resolve())
    if not required.issubset({Path(row['path']).resolve() for row in proof['files']}):
        raise ValueError('device-loop reference does not bind all numerical files')
    if proof['longcheck_job']!=host['longcheck_job'] or file_sha(host['script'])!=host['script_sha256']:
        raise ValueError('reference job or driver differs from certified body')
    c1=load(directory/'c1.json');c2=load(directory/'c2.json')
    if c1.get('pass') is not True or c1.get('positions')!=256*candidate['batch']:
        raise ValueError('reference full-model C1 did not pass')
    if c2.get('pass') is not True or c2.get('same_binary') is not True or not any(
            r.get('pass') is True and r['mode']=='L1' and r['decode_loop']==1 for r in c2['checks']):
        raise ValueError('reference did not certify L1 device-loop C2')
    actual=load(directory/'actual-execution.json')
    rows=[r for r in actual['calls'] if r['tokens']==1024 and r['executor']=='L1' and r['device_loop_used']]
    if (actual['driver_sha256']!=host['script_sha256'] or len(rows)!=1 or
            rows[0]['actual_decode_loop_calls']!=1 or rows[0]['actual_decode_host_calls']!=0):
        raise ValueError('reference did not actually execute the device loop')
    row=next(r for r in value['generation_runs'] if r['N']==1024 and r['run']==3)
    if load(path.parent/row['tokens_file'])!=load(directory/'tokens_L1_loop1.json'):
        raise ValueError('reference loop tokens differ from C1/C2-certified generation')


def validate_loop_output(candidate, path):
    if not candidate['loop']:
        return
    proof=load(candidate['numeric_reference']['path']);reference_path=Path(proof['measurement']);reference=load(reference_path)
    row=next(r for r in reference['generation_runs'] if r['N']==1024 and r['run']==3)
    expected=load(reference_path.parent/row['tokens_file'])
    path=Path(path);value=load(path)
    if any(load(path.parent/r['tokens_file'])!=[tokens[:r['N']] for tokens in expected] for r in value['generation_runs']):
        raise ValueError('device-loop timing tokens differ from independent C1/C2 reference')


def check_pair(candidate):
    runtime_contract(candidate)
    import shared_checkpoint_contract as shared
    if file_sha(shared.__file__) != candidate["shared_checkpoint_helper_sha256"]:
        raise ValueError("shared checkpoint helper changed")
    shared.validate(candidate,live=False)
    identities = {}
    for phase in ('prefill', 'decode'):
        identities[phase] = native_identity(candidate[phase])
        assert_implementation(identities[phase], candidate, phase)
    validate_loop_reference(candidate,identities)
    return identities

def execute(args):
    candidate = load(args.candidate)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if file_sha(__file__) != candidate['script_sha256']:
        raise ValueError('candidate driver changed after queue generation')
    identities = check_pair(candidate)
    runtime = runtime_contract(candidate)
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
    if (candidate['mode'],int(candidate['loop'])) not in {('L2',0),('L1',1)}:
        raise ValueError('MoE timing supports L2 host or L1 device loop')
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
    parsed = validate_native_arguments(Path(candidate.get('runtime_source',{}).get('root',candidate['snapshot_source']))/'python/tilemega/serving/r16_run.py', command[command.index('native-run')+1:])
    parsed.update(outer_system=candidate['system'], variant=candidate['variant'])
    write(output/'native-argument-contract.json', parsed)
    environment=dict(os.environ)
    if runtime is not None:
        environment.update(TILEMEGA_R16_RUNTIME_SOURCE=json.dumps(candidate['runtime_source']),PYTHONPATH=str(Path(candidate['runtime_source']['root'])/'python'),PYTHONNOUSERSITE='1')
    result = subprocess.run(command,env=environment)
    if result.returncode:
        write(output/'candidate-rejection.json', dict(metadata, exit_code=result.returncode,
            reason='official compile/runtime/numerical path rejected; no fallback was substituted'))
    raise SystemExit(result.returncode)

def validate_launch_counts(count, mode, loop, host_calls, loop_calls, host_modes):
    if loop:
        if host_calls or loop_calls != 1 or host_modes:
            raise ValueError('device loop did not use exactly one launch_steps and zero host launches')
    elif host_calls != count-1 or loop_calls or (count > 1 and set(host_modes) != {mode}):
        raise ValueError('actual Plan.launch calls disagree with selected host executor')


def execution_row(engine, result, count):
    expected=bool(engine.decode_loop and count>1)
    if bool(result.decode_loop_used)!=expected:
        raise ValueError('requested device-loop execution silently fell back to host launches')
    return {'tokens':count,'executor':'L1' if engine.decode_mode==1 else 'L2',
            'requested_device_loop':expected,'device_loop_used':bool(result.decode_loop_used)}

def native_run():
    source=json.loads(os.environ['TILEMEGA_R16_RUNTIME_SOURCE']) if 'TILEMEGA_R16_RUNTIME_SOURCE' in os.environ else None
    if source is not None:
        import r16_runtime_contract as runtime
        runtime.validate_source(source)
    from tilemega.serving import engine, r16_run, plan
    from tilemega.serving.baseline_common import guarded_main
    if source is not None:runtime.attest(source)
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
        validate_launch_counts(count,current.decode_mode,row['device_loop_used'],active['host_calls'],active['loop_calls'],active['host_modes'])
        row.update(actual_decode_host_calls=active['host_calls'],actual_decode_loop_calls=active['loop_calls'],
                   actual_decode_host_modes=sorted(active['host_modes']),
                   selected_host_kernel=('tilemega_l2_kernel' if current.decode_mode==2 else 'tilemega_l1_kernel') if not row['device_loop_used'] else None,
                   selected_device_kernel='tilemega_l1_loop_kernel' if row['device_loop_used'] else None)
        rows.append(row)
        active.clear()
        return result
    engine.ServingEngine.generate=checked
    plan.Plan.launch=launch
    plan.Plan.launch_steps=steps
    output=Path(sys.argv[sys.argv.index('--out')+1])
    try:guarded_main(r16_run.main)
    finally:
        if source is not None:write(output/'runtime-execution.json',runtime.attest(source))
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
    runtime=runtime_contract(candidate)
    if runtime is not None:runtime.validate_receipt(candidate,value,directory)
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
    for row in calls:
        validate_launch_counts(row['tokens'],1 if row['executor']=='L1' else 2,row['device_loop_used'],row['actual_decode_host_calls'],row['actual_decode_loop_calls'],row['actual_decode_host_modes'])
        if row['device_loop_used'] and row.get('selected_device_kernel')!='tilemega_l1_loop_kernel':
            raise ValueError('device-loop kernel dispatch evidence missing')
    expected = check_pair(candidate)
    validate_loop_output(candidate,path)
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
