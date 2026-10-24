#!/usr/bin/env python3
"""Measure immutable TileMega LLM controls under the user's closeout scope."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write
from run_r16_with_device_identity import require_guard

sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import verify
from tilemega.serving.baseline_common import guarded_main, validate_prompts

POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'
MODELS = {'qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_1b', 'llama3_2_3b',
          'llama3_1_8b', 'qwen3_30b_a3b'}
PROTOCOL = dict(isl=64, osl=1024, warmup=1, repeats=3, dtype='bfloat16', sampling='greedy')
AMENDMENT = dict(scope='TileMega dense25, MoE5 and five DNN E2E; frozen external baselines',
                 solver_gate='not_met', c1='not_performed', c2='not_performed',
                 hf='not_performed', full_ctest='not_performed', external_baselines='not_remeasured')


def absolute(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate(path):
    manifest = read(path)
    if (manifest.get('schema') != 1 or manifest.get('kind') != 'sm90r2_closeout_llm'
            or manifest.get('model') not in MODELS or manifest.get('batch') not in (1, 2, 4, 8, 16)
            or manifest.get('protocol') != PROTOCOL or manifest.get('e3_validated') is not False):
        raise ValueError('closeout requires the explicit unvalidated-solver LLM protocol')
    model, prompt = absolute(manifest['model_dir']), absolute(manifest['prompt_ids'])
    if sha(model / 'config.json') != manifest['model_config_sha256']:
        raise ValueError('model configuration differs from its frozen input')
    if sha(prompt) != manifest['prompt_ids_sha256'] or validate_prompts(read(prompt), manifest['batch']) != 64:
        raise ValueError('frozen ISL64 input differs')
    identities, libraries = {}, {}
    for phase in ('prefill', 'decode'):
        binding = manifest['libraries'][phase]
        library = absolute(binding['path'])
        identity = verify(library)
        if (identity['binary_sha256'] != binding['sha256']
                or identity['artifact_id'] != binding['artifact_id']
                or sha(str(library) + '.identity.json') != binding['identity_sha256']):
            raise ValueError('artifact binding changed: ' + phase)
        if identity['trace'] or any(k.startswith(('TILEMEGA_TRACE', 'TILEMEGA_PAGE_TRACE',
                'TILEMEGA_SM90_DENSE_BOUNDARY')) and str(v).strip() not in ('0', 'false')
                for k, v in identity['defines'].items()):
            raise ValueError('instrumented artifacts cannot enter closeout timing')
        plan = identity['plan']
        if (identity['nvcc_arch'] != 'sm_90a' or plan['phase'] != phase
                or plan['seq'] != (64 if phase == 'prefill' else 1)
                or plan['capacity'] < 1088 or not plan['batch_lo'] <= manifest['batch'] <= plan['batch_hi']):
            raise ValueError('compiled workload differs: ' + phase)
        if manifest.get('build_head') and identity['source']['head'] != manifest['build_head']:
            raise ValueError('artifact differs from the declared source HEAD')
        identities[phase], libraries[phase] = identity, library
    # Export filenames are display labels, not model identities; native decode
    # is named "decode.export". Shared checkpoint recipes define compatibility.
    from check_sm90r2_native_abi import host_library
    from tilemega.serving.weights import weight_recipes
    weight_recipes(*(host_library(libraries[p]) for p in ('prefill', 'decode')))
    if manifest['model'] != 'qwen3_30b_a3b':
        plan = identities['decode']['plan']
        if ('native_dense' not in plan or plan.get('native_origin') not in ('fixed', 'solver')
                or manifest['execution'] != dict(prefill='L1', decode='L1', loop=1)):
            raise ValueError('dense closeout requires the native static device-loop path')
    return manifest, model, prompt, libraries, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--fresh', type=int, default=0)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    manifest, model, prompt, libraries, identities = validate(args.manifest)
    if args.validate_only:
        print(json.dumps(dict(valid=True, model=manifest['model'], batch=manifest['batch'],
                              artifact_ids={p: i['artifact_id'] for p, i in identities.items()},
                              e3_validated=False)))
        return 0
    if args.out is None or args.fresh < 0:
        parser.error('--out and a nonnegative fresh process index are required')
    gpu = int(os.environ['TILEMEGA_DEVICE_INDEX'])
    if gpu not in (0, 1, 2, 3, 4, 7):
        raise ValueError('GPU6 excluded; GPU5 functional-only')
    require_guard(read(POLICY), gpu)
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out / 'selected-config.json'
    if receipt.exists():
        raise FileExistsError('retain prior measurement evidence')
    execution = manifest['execution']
    command = [PYTHON, '-m', 'tilemega.serving.r16_run', '--prefill', str(libraries['prefill']),
               '--decode', str(libraries['decode']), '--prompt-ids', str(prompt),
               '--batch', str(manifest['batch']), '--max-tokens', '1024', '--warmup', '1',
               '--repeats', '3', '--mode', execution['decode'], '--decode-loop', str(execution['loop']),
               '--prefill-mode', execution['prefill'], '--round-index', str(args.fresh),
               '--out', str(args.out / 'measurements')]
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'python'))
    if manifest.get('checkpoint'):
        checkpoint = absolute(manifest['checkpoint']['path'])
        info = read(checkpoint / 'random_checkpoint.json')
        if (info.get('state') != 'live' or info['checkpoint_sha256'] != manifest['checkpoint']['sha256']):
            raise ValueError('shared random checkpoint is not the declared live checkpoint')
        environment['TILEMEGA_RANDOM_CHECKPOINT'] = str(checkpoint)
    else:
        command = [PYTHON, '-m', 'tilemega.serving.random_checkpoint', '--config', str(model),
                   '--out', str(args.out / 'checkpoint'), '--seed', '0', '--max-gib',
                   '80' if manifest['model'] == 'qwen3_30b_a3b' else '32', '--', *command]
    record = dict(manifest=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest),
                  driver_sha256=sha(__file__), model=manifest['model'], batch=manifest['batch'],
                  fresh_process=args.fresh, stage='measure', protocol=PROTOCOL,
                  selection=manifest['selection'], e3_validated=False, user_amendment=AMENDMENT,
                  artifact_ids={p: i['artifact_id'] for p, i in identities.items()},
                  binary_sha256={p: i['binary_sha256'] for p, i in identities.items()},
                  source_heads={p: i['source']['head'] for p, i in identities.items()},
                  command=command, started=time.time())
    write(receipt, record)
    status = subprocess.call(command, cwd=ROOT, env=environment)
    write(args.out / 'execution.json', dict(**record, exit_code=status, ended=time.time()))
    return status


if __name__ == '__main__':
    guarded_main(lambda: sys.exit(main()))
