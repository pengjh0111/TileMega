#!/usr/bin/env python3
"""Prepare guarded full-request Hopper selection; never enqueue GPU work."""
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

from r16_measurement_validation import (completed_guard, file_sha, native_identity,
    validate_llm, validate_native_repetitions)

MODELS = ('llama3_2_1b', 'qwen3_0_6b', 'qwen3_1_7b',
          'llama3_2_3b', 'llama3_1_8b', 'qwen3_30b_a3b')
ARMS = {'mma': 'TM-noWGMMA', 'sm80': 'TM-sm80',
        'h1': 'TM-H1', 'h1h4': 'TM-H1H4'}
CLASSES = ('qkv', 'o', 'gate-up', 'down', 'lm-head')
TERMINAL = {'done', 'failed', 'skipped', 'not_run', 'cancelled'}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def gemm_count(config):
    return (5 if config['model_type'] == 'qwen3_moe' else 4)*config['num_hidden_layers']+1


def shared_layout(manifest):
    recipes = {}
    for buffer in manifest['buffers']:
        recipe = buffer.get('recipe')
        if buffer.get('dtype') != 'bf16' or not recipe:
            continue
        if recipe['kind'] == 'tile_pages':
            recipe = recipe['source']
        recipes[json.dumps(recipe, sort_keys=True)] = {'dtype': 'bf16', 'recipe': {
            'kind': 'tile_pages', 'tile_n': 128, 'tile_k': 64, 'source': recipe}}
    if not recipes:
        raise ValueError('seed manifest has no BF16 weight recipes')
    return {'buffers': list(recipes.values())}


def assert_implementation(identity, candidate, phase):
    if identity['source']['root'] != candidate['snapshot_source']:
        raise ValueError('artifact was compiled from a different source snapshot')
    if identity['source']['head'] != candidate['snapshot_head']:
        raise ValueError('artifact source HEAD differs from frozen snapshot')
    if identity['source']['compiler_sha256'] != candidate['compiler_sha256']:
        raise ValueError('artifact compiler differs from frozen CLI')
    # Earlier sidecars used the compiler target in both fields. Their logical
    # architecture is still independently recorded by the complete TargetSpec.
    if (identity.get('arch') not in ('sm_90', 'sm_90a') or identity.get('nvcc_arch') != 'sm_90a'
            or identity.get('plan', {}).get('target_spec', {}).get('arch_tag') != 'sm_90'):
        raise ValueError('candidate requires explicit sm_90a compilation')
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
        if any(row['impl'] != requested for row in rows):
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
    if any((g['tile_n'],g['tile_k'])!=(128,64) for g in plan['gemms']):
        raise ValueError('actual GEMM geometry differs from the shared T128/K64 layout')
    if phase == 'decode':
        for coordinate, macro in (('activation_tma', 'TILEMEGA_ACTIVATION_TMA'),
                                  ('qkv_epilogue', 'TILEMEGA_QKV_EPILOGUE')):
            wanted = int(candidate.get(coordinate, 0))
            if int(plan.get(coordinate, 0)) != wanted or int(identity.get('defines', {}).get(macro, 0)) != wanted:
                raise ValueError(f'{coordinate} differs from the requested candidate')
        expected = 'wgmma' if candidate['arm'] == 'h1h4' else 'mma16'
        attention = identity['implementations']['attention']
        if attention['impl'] != expected or attention['page_layout'] != ('uniform' if expected == 'wgmma' else 'chunk16'):
            raise ValueError('requested decode attention silently changed or fell back')
        if requested == 'wgmma' and not all(identity['caps'].get(name) is True for name in ('wgmma', 'tma', 'bulk_copy')):
            raise ValueError('Hopper capability evidence is absent')
    if identity.get('trace') is not False:
        raise ValueError('instrumented binary cannot enter performance selection')


def check_pair(candidate):
    identities = {}
    for phase in ('prefill', 'decode'):
        identities[phase] = native_identity(candidate[phase])
        assert_implementation(identities[phase], candidate, phase)
    return identities


def environment(root, source, gpu, pid_mode):
    return {'CUDA_VISIBLE_DEVICES': str(gpu), 'TILEMEGA_DEVICE_INDEX': str(gpu),
            'TILEMEGA_NUMA_NODE': str(0 if gpu < 4 else 1),
            'TILEMEGA_GPU_PID_MODE': pid_mode, 'PYTHONPATH': str(source/'python'),
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '8'}


def job(candidate, name, command, *, gpu=True, timing=False, after=(), priority=25, timeout=1800):
    node = 0 if candidate['gpu'] < 4 else 1
    binding = ['numactl', '--cpunodebind='+str(node)]
    if not candidate['cpu_bind_only']:
        binding += ['--membind='+str(node)]
    return {'name': name, 'command': binding+list(map(str, command)), 'gpu': gpu,
            'timing': timing, 'after': list(after), 'priority': candidate.get('priority_base',0)+priority,
            'timeout_s': timeout, 'needs_free_mib': candidate['needs_free_mib'] if timing else 2048,
            'cwd': candidate['root'], 'env': candidate['env']}


def compile_command(candidate, phase, output, *, bootstrap=False):
    command = [candidate['compiler'], 'compile', candidate['exports'][phase], str(output),
               '--serving', phase, '--emit', 'serving', '--batch', str(candidate['batch']),
               '--runtime-target', candidate['target'], '--pg', 'off',
               '--nonpaged-weight-layout', 'tiled', '--deferred-norm', '0',
               '--resource-probes', '0', '--capacity', '1088', '--prompt-len', '64',
               '--arch-paths', 'sm80' if candidate['arm'] == 'sm80' else 'auto']
    if not bootstrap:
        command += ['--seed-arch', 'native']
    if phase == 'decode':
        command += ['--past-range', '64:1086']
    if not bootstrap:
        command += ['--shared-weight-layout', candidate['layout']]
        if phase == 'decode':
            impl = 'wgmma' if candidate['arm'] in ('h1', 'h1h4') else 'mma16'
            for name in CLASSES:
                command += ['--gemm-impl-'+name, impl]
            if candidate['arm'] == 'h1h4':
                command += ['--attention-impl', 'wgmma', '--attention-page', 'uniform']
            if candidate['moe'] and impl == 'wgmma':
                command += ['--moe-activation-bulk', '1']
            for coordinate in ('activation_tma', 'qkv_epilogue'):
                if candidate.get(coordinate, 0):
                    command += ['--'+coordinate.replace('_', '-'), '1']
    return command


def run_command(candidate, stage, round_index, output):
    return [candidate['python'], candidate['script'], 'execute', '--candidate',
            candidate['file'], '--stage', stage, '--round-index', str(round_index), '--out', str(output)]


def prepare(args):
    root = args.root.resolve()
    source = args.snapshot.resolve()
    receipt = load(str(source)+'.json')
    compiler = Path(receipt['cli'])
    if receipt.get('build_returncode') != 0 or file_sha(compiler) != receipt['cli_sha256']:
        raise ValueError('frozen compiler build is absent or changed')
    if '"--seed-arch"' not in (source/'tools/commands/compile.cpp').read_text():
        raise ValueError('snapshot predates the official native seed binding option; use its completed successor')
    for coordinate in ('activation_tma', 'qkv_epilogue'):
        flag = '--'+coordinate.replace('_', '-')
        if getattr(args, coordinate, 0) and '"'+flag+'"' not in (source/'tools/commands/compile.cpp').read_text():
            raise ValueError(f'snapshot does not support {flag}')
    if not 0 <= args.gpu < 8 or not 0 <= args.moe_gpu < 8:
        raise ValueError('GPU index must identify one of the eight physical cards')
    out = args.out.resolve()
    if not out.is_relative_to(root/'runs/r16'):
        raise ValueError('generated queues and measurements must remain under runs/r16')
    if (out/'plan.json').exists():
        raise ValueError('run directory already contains a plan; retain it and choose a new run-id')
    sys.path.insert(0, str(source/'python'))
    from tilemega.serving.integrated_selection import coverage_order, light_candidates
    queues, cells, rejected = {}, [], []
    for model_index, model in enumerate(args.models):
        config_directory = root/'runs/r16/models'/model
        try:
            config = load(config_directory/'config.json')
            exports = {phase: str(root/'runs/r16/exports'/model/phase/'export.json') for phase in ('prefill', 'decode')}
            export_sha = {phase: file_sha(path) for phase, path in exports.items()}
        except (OSError, ValueError, KeyError) as error:
            rejected.append({'model': model, 'stage': 'prepare', 'error': str(error)})
            continue
        gpu = args.moe_gpu if config['model_type'] == 'qwen3_moe' else args.gpu
        for batch in args.batches:
            directory = out/model/f'B{batch}'
            prefix = f'rf2_{args.run_id}_{model}_B{batch}'
            base = dict(root=str(root), snapshot_source=str(source), snapshot_head=receipt['snapshot_head'],
                compiler=str(compiler), compiler_sha256=receipt['cli_sha256'], target=str(source/'configs/targets/sm_90_h800.json'),
                model=model, config=str(config_directory), config_sha256=file_sha(config_directory/'config.json'),
                exports=exports, export_sha256=export_sha, gemm_count=gemm_count(config), batch=batch,
                priority_base=model_index*100, seed_arch='native',
                gpu=gpu, env=environment(root, source, gpu, args.pid_mode), cpu_bind_only=args.cpu_bind_only,
                needs_free_mib=74000 if config['model_type']=='qwen3_moe' else (42000 if config['hidden_size']>=4096 else 24000),
                python=str(args.python.absolute()), script=str(Path(__file__).resolve()), seed=args.seed,
                script_sha256=file_sha(__file__),
                moe=config['model_type']=='qwen3_moe', layout=str(directory/'shared-t128x64.json'),
                prompt_ids=str(root/'configs/prompts'/('qwen_isl64.json' if model.startswith('qwen') else 'llama3_isl64.json')))
            queue = queues.setdefault(gpu, [])
            seeds = [root/'runs/r16/plans/seed'/model/f'B{b}'/'decode.so.plan.json' for b in (batch, 1, 16)]
            seed = next((path for path in seeds if path.is_file()), None)
            dependencies = []
            if seed is None:
                bootstrap = dict(base, arm='mma')
                library = directory/'bootstrap-seed.so'
                seed = Path(str(library)+'.plan.json')
                name = prefix+'_bootstrap_seed'
                queue.append(job(base, name, compile_command(bootstrap, 'decode', library, bootstrap=True)))
                dependencies = [name]
            layout_name = prefix+'_layout'
            queue.append(job(base, layout_name, [base['python'], base['script'], 'layout', '--seed-manifest', seed,
                '--count', base['gemm_count'], '--out', base['layout']], gpu=False, after=dependencies))
            prefill_names = {}
            prefill_paths = {}
            for transport in ('auto', 'sm80'):
                arm = 'mma' if transport == 'auto' else 'sm80'
                library = directory/f'prefill-{transport}.so'
                name = prefix+'_prefill_'+transport
                queue.append(job(base, name, compile_command(dict(base, arm=arm), 'prefill', library), after=[layout_name]))
                prefill_names[transport], prefill_paths[transport] = name, str(library)
            choices = []
            for arm in ARMS:
                transport = 'sm80' if arm == 'sm80' else 'auto'
                decode = directory/f'decode-{arm}.so'
                build_name, smoke_name = prefix+'_decode_'+arm, prefix+'_smoke_'+arm
                candidate = dict(base, arm=arm, prefill=prefill_paths[transport], decode=str(decode),
                                 system=ARMS[arm], family=arm, mode='L2', loop=0,
                                 variant={'compute':arm}, base_variant=arm=='mma')
                candidate['activation_tma'] = int(getattr(args, 'activation_tma', 0) and arm in ('h1', 'h1h4'))
                candidate['qkv_epilogue'] = int(getattr(args, 'qkv_epilogue', 0) and arm == 'h1h4')
                for coordinate in ('activation_tma', 'qkv_epilogue'):
                    candidate['variant'][coordinate] = candidate[coordinate]
                candidate['file'] = str(directory/(arm+'-L2.json'))
                write(candidate['file'], candidate)
                queue.append(job(base, build_name, compile_command(candidate, 'decode', decode), after=[layout_name]))
                smoke = job(base, smoke_name, run_command(candidate, 'smoke', 0, '{attempt_dir}/measurements'),
                    after=[build_name, prefill_names[transport]], timeout=1800)
                smoke['needs_free_mib'] = base['needs_free_mib']
                queue.append(smoke)
                for mode, loop in (('L2', 0), ('L1', 1)):
                    item = dict(candidate, mode=mode, loop=loop, id=arm+'-'+mode+f'-loop{loop}',
                                file=str(directory/(arm+'-'+mode+f'-loop{loop}.json')))
                    item['pilot_job'] = prefix+'_pilot_'+item['id']
                    item['smoke_job'] = smoke_name
                    write(item['file'], item)
                    choices.append(item)
            candidates = light_candidates(coverage_order(choices), 8)
            for i, candidate in enumerate(candidates):
                queue.append(job(candidate, candidate['pilot_job'], run_command(candidate, 'pilot', 0, '{attempt_dir}/measurements'),
                    timing=True, after=[candidate['smoke_job']], priority=40+i, timeout=3600))
            cells.append(dict(model=model, batch=batch, gpu=gpu, candidates=candidates))
    for gpu, queue in queues.items():
        write(out/f'gpu{gpu}/queue_prepare_and_pilot.json', queue)
    write(out/'plan.json', dict(schema=1, evidence='queue_generation_only', root=str(root), run_id=args.run_id,
        snapshot=str(source), snapshot_head=receipt['snapshot_head'], compiler_sha256=receipt['cli_sha256'],
        objective='full request E2E seconds; TPOT/TTFT/throughput also reported',
        protocol={'isl':64, 'osl':1024, 'warmup':1, 'repeats':3, 'pilot_rounds':1, 'final_fresh_rounds':3, 'maximum_candidates':8},
        selection_limit='bounded body/executor candidate screen; no exhaustive geometry or RF2 calibrated solver claim',
        reuse_policy='existing accepted jobs with identical names and identities remain done; older snapshot binaries are not substituted',
        cells=cells, rejected=rejected, enqueued=False))
    print(json.dumps({'cells':len(cells), 'jobs':sum(map(len, queues.values())), 'rejected':rejected, 'out':str(out)}))


def execute(args):
    candidate = load(args.candidate)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if file_sha(__file__) != candidate['script_sha256']:
        raise ValueError('candidate driver changed after queue generation')
    identities = check_pair(candidate)
    metadata = dict(stage=args.stage, candidate=candidate['id'] if 'id' in candidate else candidate['arm'],
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
        '--prompt-ids', candidate['prompt_ids'], '--batch', str(candidate['batch']), '--system-label', candidate['system'],
        '--max-tokens', '16' if args.stage=='smoke' else '1024', '--warmup', '1', '--repeats', '3',
        '--mode', candidate['mode'], '--decode-loop', str(candidate['loop']), '--round-index', str(args.round_index),
        '--out', str(output)]
    if args.stage == 'smoke':
        command += ['--check-executors', '--hf-check']
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
    from tilemega.serving import engine, r16_run
    from tilemega.serving.baseline_common import guarded_main
    original=engine.ServingEngine.generate
    rows=[]
    def checked(current,prompt_ids,new_tokens=None):
        result=original(current,prompt_ids,new_tokens)
        rows.append(execution_row(current,result,new_tokens or current.max_new_tokens))
        return result
    engine.ServingEngine.generate=checked
    output=Path(sys.argv[sys.argv.index('--out')+1])
    try:guarded_main(r16_run.main)
    finally:
        engine.ServingEngine.generate=original
        write(output/'actual-execution.json',{'evidence':'verified runtime Generation.decode_loop_used',
            'driver_sha256':file_sha(__file__),'calls':rows})


def states(root):
    result = {}
    for path in (root/'runs/r16/scheduler').glob('gpu*/state.json'):
        for name, value in load(path).items():
            if name in result:
                raise ValueError('job name occurs in multiple card schedulers: '+name)
            result[name] = value
    return result


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


def validate_smoke(root, state, candidate):
    record=state[candidate['smoke_job']]
    directory=Path(record['out'])
    if not directory.is_absolute():directory=root/directory
    completed_guard(directory,record)
    output=directory/'measurements'
    validate_native_repetitions(output/f"B{candidate['batch']}"/'measurements.json')
    evidence=load(output/'candidate-evidence.json')
    expected=check_pair(candidate)
    if evidence['artifact_ids']!={phase:x['artifact_id'] for phase,x in expected.items()}:
        raise ValueError('smoke certified another binary pair')
    c1,c2=load(output/'c1.json'),load(output/'c2.json')
    if c1.get('pass') is not True or c1.get('positions')!=16*candidate['batch']:
        raise ValueError('shared-checkpoint 16-token HF check did not pass')
    checks=c2.get('checks',[])
    if c2.get('pass') is not True or c2.get('same_binary') is not True or {
        (row['mode'],int(row['decode_loop'])) for row in checks if row.get('pass') is True
    }!={('L1',0),('L2',0),('L1',1)}:
        raise ValueError('same-binary executor token checks did not all pass')


def finalists(args):
    plan = load(args.plan)
    root = Path(plan['root'])
    state = states(root)
    output = args.plan.parent
    queues, decisions = {}, []
    for cell in plan['cells']:
        valid, rejected, pending = [], [], []
        for candidate in cell['candidates']:
            name = candidate['pilot_job']
            status = state.get(name, {}).get('status')
            if status not in TERMINAL:
                pending.append(name)
                continue
            try:
                validate_smoke(root,state,candidate)
                value = measurement(root, state, name, candidate)
                valid.append((value['e2e_seconds'], candidate['id'], candidate, value))
            except (OSError, ValueError, KeyError, TypeError) as error:
                rejected.append({'candidate':candidate['id'], 'job':name, 'error':str(error), 'status':status})
        decision = dict(model=cell['model'], batch=cell['batch'], rejected=rejected, pending=pending,
                        pilot_measurements=[entry[3] for entry in sorted(valid)])
        if pending or not valid:
            decision['status'] = 'incomplete' if pending else 'all_candidates_rejected'
            decisions.append(decision)
            continue
        if len({entry[3]['checkpoint_sha256'] for entry in valid}) != 1 or len({json.dumps(entry[3]['versions'],sort_keys=True) for entry in valid}) != 1:
            decision['status']='candidate_checkpoint_or_versions_differ'
            decisions.append(decision)
            continue
        selected = [entry[2] for entry in sorted(valid)[:3]]
        decision.update(status='finalists_ready', finalists=selected)
        for round_index in range(3):
            for rank, candidate in enumerate(selected[round_index%len(selected):]+selected[:round_index%len(selected)]):
                name = f"rf2_{plan['run_id']}_{cell['model']}_B{cell['batch']}_final{round_index}_{candidate['id']}"
                queue = queues.setdefault(candidate['gpu'], [])
                queue.append(job(candidate, name, run_command(candidate, 'final', round_index, '{attempt_dir}/measurements'),
                    timing=True, priority=70+round_index*3+rank, timeout=3600))
        decisions.append(decision)
    for gpu, queue in queues.items():
        write(output/f'gpu{gpu}/queue_finalists.json', queue)
    write(output/'selection-stage.json', {'evidence':'pilot measured; final queues not enqueued', 'cells':decisions})
    print(json.dumps({'ready_cells':sum(x['status']=='finalists_ready' for x in decisions),
                      'final_jobs':sum(map(len,queues.values())), 'states':{x['model']+'/B'+str(x['batch']):x['status'] for x in decisions}}))


def report(args):
    plan = load(args.plan)
    stage = load(args.plan.parent/'selection-stage.json')
    root = Path(plan['root'])
    state = states(root)
    cells = []
    for decision in stage['cells']:
        rows, errors, pending = [], [], []
        for candidate in decision.get('finalists', []):
            samples = []
            for round_index in range(3):
                name=f"rf2_{plan['run_id']}_{decision['model']}_B{decision['batch']}_final{round_index}_{candidate['id']}"
                try:
                    value=measurement(root,state,name,candidate)
                    if value['pid'] in {sample['pid'] for sample in samples}:
                        raise ValueError('confirmation rounds do not use fresh processes')
                    samples.append(value)
                except (OSError,ValueError,KeyError,TypeError) as error:
                    errors.append({'job':name,'error':str(error)})
                    if state.get(name,{}).get('status') not in TERMINAL:
                        pending.append(name)
            if len(samples)==3:
                if len({x['checkpoint_sha256'] for x in samples})!=1 or len({json.dumps(x['versions'],sort_keys=True) for x in samples})!=1:
                    errors.append({'candidate':candidate['id'],'error':'confirmation checkpoint or versions differ'})
                    continue
                rows.append(dict(candidate=candidate['id'], samples=samples,
                    **{field:statistics.median(sample[field] for sample in samples)
                       for field in ('e2e_seconds','ttft_seconds','tpot_seconds','tokens_per_second')}))
        if rows and (len({s['checkpoint_sha256'] for row in rows for s in row['samples']})!=1 or
                     len({json.dumps(s['versions'],sort_keys=True) for row in rows for s in row['samples']})!=1):
            errors.append({'error':'finalists do not share checkpoint identity and installed versions'})
            rows=[]
        passed=bool(rows) and not pending
        cells.append(dict(model=decision['model'],batch=decision['batch'],status='complete' if passed else 'incomplete',
            candidates=rows,errors=errors,pending=pending,selected=min(rows,key=lambda row:(row['e2e_seconds'],row['candidate']))['candidate'] if passed else None,
            default_promoted=False,limit='selection does not establish all-cell non-regression, true pretrained C-1, or strict GPU ownership'))
    write(args.plan.parent/'selection-report.json', {'evidence':'full-request measured selection', 'cells':cells})
    print(json.dumps({'complete_cells':sum(c['status']=='complete' for c in cells),'cells':len(cells)}))


def main():
    if sys.argv[1:2]==['native-run']:
        sys.argv.pop(1)
        return native_run()
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--python',type=Path,default=Path('/workspace/llm-stack/envs/vllm/bin/python'))
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--batches',nargs='+',type=int,choices=(1,2,4,8,16),default=[1,16])
    p.add_argument('--activation-tma',type=int,choices=(0,1),default=0)
    p.add_argument('--qkv-epilogue',type=int,choices=(0,1),default=0)
    p.add_argument('--gpu',type=int,default=6)
    p.add_argument('--moe-gpu',type=int,default=3)
    p.add_argument('--pid-mode',choices=('strict','namespace_limited'),default='strict')
    p.add_argument('--cpu-bind-only',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    for name in ('finalists','report'):
        p=sub.add_parser(name);p.add_argument('--plan',type=Path,required=True)
    p=sub.add_parser('layout');p.add_argument('--seed-manifest',type=Path,required=True)
    p.add_argument('--count',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('execute');p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','pilot','final'),required=True)
    p.add_argument('--round-index',type=int,default=0);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='layout':
        manifest=load(args.seed_manifest)
        if len(manifest['gemms'])!=args.count:raise ValueError('seed is not the expected full model')
        write(args.out,shared_layout(manifest))
    elif args.command=='execute':
        try:execute(args)
        except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError) as error:
            write(args.out/'candidate-rejection.json', {'evidence':'verified rejection','error':str(error),
                'candidate':str(args.candidate),'stage':args.stage,'fallback_substituted':False})
            raise SystemExit(3) from error
    elif args.command=='finalists':finalists(args)
    else:report(args)


if __name__=='__main__':
    main()
