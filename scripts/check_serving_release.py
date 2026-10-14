#!/usr/bin/env python3
"""Freeze selected serving arms and gate comparisons on full-request correctness.

All GPU actions require the scheduler guard. Recipes contain local paths only;
this driver never fetches checkpoints or promotes an unvalidated selection.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

from tilemega.bench.arms import check_binaries, register, sha256
from tilemega.bench.anchor import command
from tilemega.serving.workload import validate_prompt_ids


def checkpoint_files(model):
    model = Path(model)
    index = model/'model.safetensors.index.json'
    names = (set(json.loads(index.read_text())['weight_map'].values()) if index.exists()
             else {'model.safetensors'})
    records = []
    for name in sorted(names):
        path = (model/name).resolve()
        if not path.is_relative_to(model.resolve()) or not path.is_file():
            raise ValueError('missing local checkpoint shard: ' + str(path))
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if length > path.stat().st_size - 8:
                raise ValueError('truncated safetensors header: ' + str(path))
            header = json.loads(stream.read(length))
        end = max((v['data_offsets'][1] for k,v in header.items() if k != '__metadata__'), default=0)
        if path.stat().st_size != 8 + length + end:
            raise ValueError('incomplete safetensors data: ' + str(path))
        records.append(dict(path=str(path), bytes=path.stat().st_size))
    return records


def step_bytes(config, workload):
    h, inter = config['hidden_size'], config['intermediate_size']
    layers, heads = config['num_hidden_layers'], config['num_attention_heads']
    kv = config.get('num_key_value_heads', heads)
    d = config.get('head_dim', h//heads)
    weights = layers*(h*(heads+2*kv)*d + heads*d*h + 3*h*inter) + h*config['vocab_size']
    # Decode launches visit ISL through ISL+OSL-2; the first output is prefill.
    mean_past = workload['isl'] + (workload['osl']-2)/2
    return 2*weights + 4*workload['batch']*layers*kv*d*(mean_past+1)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def freeze(recipe, out):
    config = json.loads(Path(recipe['config']).read_text())
    workload = recipe['workload']
    files = checkpoint_files(recipe['model_path'])
    validate_prompt_ids(json.loads(Path(recipe['prompt_ids']).read_text()), workload['batch'], workload['isl'])
    plans = json.loads(Path(recipe['plans']).read_text())[str(workload['batch'])]
    arms = []
    if recipe.get('reference'):
        arms.append(register(dict(recipe['reference'], label='R14_control', kind='tm',
                                  root=recipe['root'], python=recipe['python'],
                                  mode='L1', decode_loop=0, prefill_mode='L1')))
    arms.append(register(dict(label='R15_selected', kind='tm', root=recipe['root'],
        python=recipe['python'], prefill=plans['prefill'], decode=plans['decode'],
        mode='auto', decode_loop='auto', prefill_mode='auto')))
    selected = Path(plans['decode']+'.serving.json')
    if not selected.is_file():
        raise ValueError('selected plan has no execution sidecar')
    arms.append(dict(label='vllm', kind='vllm', root=recipe['root'], python=recipe['vllm_python']))
    targets = list(Path(recipe['target_dir']).glob('*.json'))
    if len(targets) != 1:
        raise ValueError('expected exactly one calibrated target')
    target = json.loads(targets[0].read_text())
    bandwidth = target['calibration_by_dtype']['bf16']['pipelines']['dram_gbps']
    model = json.loads((Path(recipe['model_path'])/'config.json').read_text())
    data = dict(measurement_policy=recipe['policy'], cells={recipe['cell']:dict(
        model_path=recipe['model_path'], prompt_ids=recipe['prompt_ids'], workload=workload,
        step_bytes=step_bytes(model,workload), dram_gbps=bandwidth, arms=arms)})
    write(out/'arms.json',data)
    write(out/'inputs.json',dict(recipe=recipe, config=config, checkpoint_files=files,
        plans_sha256=sha256(recipe['plans']), sidecar=str(selected), sidecar_sha256=sha256(selected),
        target=str(targets[0]), target_sha256=sha256(targets[0]),
        byte_definition='BF16 linear weights plus historical/current KV; excludes activations and norm vectors',
        arms_sha256=sha256(out/'arms.json')))
    print('PASS frozen release arms ' + recipe['cell'],flush=True)


def execute(cmd, folder, label, env=None):
    write(folder/(label+'.command.json'),cmd)
    with (folder/(label+'.log')).open('w') as stream:
        code = subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
    if code == 75:
        raise SystemExit(75)
    return code


def validate_inputs(out):
    record = json.loads((out/'inputs.json').read_text())
    if (sha256(out/'arms.json') != record['arms_sha256'] or
            sha256(record['sidecar']) != record['sidecar_sha256'] or
            sha256(record['recipe']['plans']) != record['plans_sha256'] or
            sha256(record['target']) != record['target_sha256']):
        raise ValueError('release inputs changed after registration')
    data = json.loads((out/'arms.json').read_text())
    for cell in data['cells'].values():
        for arm in cell['arms']: check_binaries(arm)
    return record, data


def correctness(out, attempt):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('correctness requires the scheduler GPU guard')
    record, data = validate_inputs(out)
    recipe = record['recipe']; cell = data['cells'][recipe['cell']]
    workload = cell['workload']; attempt.mkdir(parents=True, exist_ok=True)
    baseline = next(arm for arm in cell['arms'] if arm['kind']=='vllm')
    cmd, env = command(baseline,cell,attempt/'vllm',data['measurement_policy'])
    # Timings from this reference generation are not used for performance.
    code = execute(cmd,attempt,'vllm_generate',env)
    if code: raise RuntimeError('vLLM reference generation failed')
    generated = attempt/'vllm'/('B'+str(workload['batch']))/f"tokens_N{workload['osl']}_run1.json"
    hf = [recipe['python'],'-m','tilemega.serving.hf_check','--model',cell['model_path'],
          '--prompt-ids',cell['prompt_ids'],'--skip-free-greedy']
    vllm_metrics = attempt/'hf_vllm.json'
    code = execute([*hf,'--generated',str(generated),'--out',str(vllm_metrics)],attempt,'hf_vllm')
    # An absolute vLLM gate failure supplies the preregistered relative gate;
    # a process failure without a valid metrics receipt never does.
    reference = json.loads(vllm_metrics.read_text())
    if code not in (0,1) or reference['positions'] != workload['batch']*workload['osl']:
        raise RuntimeError('invalid HF reference receipt')
    results = {}
    for arm in cell['arms']:
        if arm['kind'] != 'tm':continue
        folder = attempt/arm['label']; folder.mkdir(parents=True,exist_ok=True)
        cmd = [recipe['python'],'-m','tilemega.serving.check_modes','--model',cell['model_path'],
            '--prefill-so',arm['prefill'],'--decode-so',arm['decode'],'--prompt-ids',cell['prompt_ids'],
            '--batch',str(workload['batch']),'--steps',str(workload['osl']),'--out',str(folder)]
        mode_code = execute(cmd,attempt,'modes_'+arm['label'])
        token_path=folder/'tokens_L1_separate.json'
        hf_code = None
        if token_path.exists():
            hf_code = execute([*hf,'--generated',str(token_path),'--out',str(folder/'hf.json'),
                '--vllm-metrics',str(vllm_metrics)],attempt,'hf_'+arm['label'])
        mode = json.loads((folder/'mode_check.json').read_text()) if (folder/'mode_check.json').exists() else {}
        results[arm['label']] = dict(c2=mode_code==0 and mode.get('pass',False),c1=hf_code==0)
    validate_inputs(out)
    passed=all(r['c1'] and r['c2'] for r in results.values())
    write(attempt/'correctness.json',dict(cell=recipe['cell'],workload=workload,
        results=results,timing_eligible=False,**{'pass':passed}))
    print(json.dumps(results),flush=True)
    if not passed:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','correctness'))
    parser.add_argument('--recipe',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--attempt',type=Path)
    args=parser.parse_args()
    if args.action=='freeze':
        if not args.recipe:parser.error('freeze requires --recipe')
        freeze(json.loads(args.recipe.read_text()),args.out)
    else:
        if not args.attempt:parser.error('correctness requires --attempt')
        correctness(args.out,args.attempt)


if __name__=='__main__':main()
