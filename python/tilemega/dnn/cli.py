"""Build and execute upstream DNN forward plans with correctness receipts."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from tilemega.cache import atomic_json, file_sha
from tilemega.fingerprint import ROOT


@contextmanager
def gpu_lock():
    path=Path(os.environ.get('TILEMEGA_GPU_LOCK','/root/r14_work/gpu.lock'))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        inode=os.fstat(stream.fileno())
        # A scheduler's flock descriptor can survive exec. Reuse its open
        # description: opening and locking the same path again would deadlock.
        for name in os.listdir('/proc/self/fd'):
            fd=int(name)
            if fd==stream.fileno():
                continue
            try:
                info=os.fstat(fd)
                if (info.st_dev,info.st_ino)!=(inode.st_dev,inode.st_ino):
                    continue
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:
                continue
            yield
            return
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def read_config(path):
    config = json.loads(Path(path).read_text())
    name = config.get('model', {}).get('name')
    if name not in ('resnet18', 'mbv1', 'mbv2', 'bert', 'nafnet'):
        raise ValueError('model.name must identify one of the five DM-1 DNNs')
    config['workload'] = dict(dict(batch=[1]), **config.get('workload', {}))
    batches = config['workload']['batch']
    if not batches or len(set(batches)) != len(batches) or any(
            type(batch) is not int or not 1 <= batch <= 64 for batch in batches):
        raise ValueError('workload.batch must contain distinct integers in [1,64]')
    config['features'] = dict(dict(pg='l2', forward_executor='L2', reuse='auto',deferred_ln='auto',dwpw_fuse='auto',global_la='auto'),
                              **config.get('features', {}))
    unknown=set(config['features'])-{'pg','forward_executor','reuse','deferred_ln','dwpw_fuse',
        'global_la','nonpaged_la','paged_la','paged_la_splitk'}
    if unknown:
        raise ValueError('unsupported DNN features: '+', '.join(sorted(unknown)))
    for key, values in dict(pg=('off', 'l2', 'pages', 'auto'),
            forward_executor=('L1', 'L2'), reuse=('auto', 'none', 'greedy', 'l2'),
            deferred_ln=('auto','0'), dwpw_fuse=('auto','0'),
            global_la=('auto','0'), nonpaged_la=(0,1), paged_la=(0,1),
            paged_la_splitk=(0,1)).items():
        if key in config['features'] and config['features'][key] not in values:
            raise ValueError('invalid features.' + key)
    config['solver'] = dict(dict(passes=2, time_budget_s=1800, jobs=1),
                            **config.get('solver', {}))
    if config['solver']['passes'] not in (1, 2, 3) or config['solver']['jobs'] < 1 or \
            config['solver']['time_budget_s'] <= 0:
        raise ValueError('invalid solver passes, jobs or budget')
    config['device'] = dict(dict(cache_dir='~/.cache/tilemega-dm'), **config.get('device', {}))
    config['output'] = dict(dict(dir='runs/dm1-' + name), **config.get('output', {}))
    config['target'] = config.get('target', str(ROOT / 'configs/targets/sm_89.json'))
    return config


def model_export(config, out):
    from .export import export, DEFAULT_NAFNET
    model = config['model']
    label = model['name'] + ('_masked' if model.get('attention_mask') else '')
    if model.get('export'):
        directory = Path(model['export']).expanduser().resolve()
    else:
        directory = out / 'exports' / label
        if not (directory / 'manifest.json').is_file():
            export(model['name'], out / 'exports', out / 'fixtures',
                mask=model.get('attention_mask', False),
                nafnet_weights=Path(model.get('nafnet_weights', DEFAULT_NAFNET)),
                checkpoint=model.get('checkpoint'))
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['model'] != label or not manifest.get('accuracy_eligible'):
        raise ValueError('export must be the requested upstream model with pretrained weights')
    for name, expected in manifest['artifacts'].items():
        if file_sha(directory / name) != expected:
            raise ValueError('upstream export differs from its manifest: ' + name)
    return directory


def build(config, out, compiler):
    import torch
    from tilemega.export_bridge import serialize
    from tilemega.build.identity import source_snapshot, generate
    export = model_export(config, out)
    archive = export / 'exported_program.pt2'
    program = torch.export.load(archive)
    plans = []
    for batch in config['workload']['batch']:
        directory = out / f'B{batch}'
        directory.mkdir(parents=True, exist_ok=True)
        library, bridge = directory / 'plan.so', directory / 'export.json'
        if library.exists():
            raise FileExistsError('refuse to overwrite a built artifact: ' + str(library))
        symbols = {str(node.meta['val'].shape[0]): batch for node in program.graph.nodes
            if node.op == 'placeholder' and isinstance(node.meta.get('val'), torch.Tensor)
            and node.meta['val'].ndim and not isinstance(node.meta['val'].shape[0], int)}
        atomic_json(bridge, serialize(program, shape_bindings=symbols))
        command = [str(compiler), 'compile', str(bridge), str(library),
            '--emit', 'serving', '--serving', 'forward', '--frontend', 'dnn',
            '--batch', str(batch), '--pg', config['features']['pg'],
            '--reuse', config['features']['reuse'], '--solve', config['target'],
            '--deferred-ln', config['features']['deferred_ln'],
            '--dwpw-fuse', config['features']['dwpw_fuse'],
            '--runtime-target', config['target'], '--selection', 'predicted',
            '--search-passes', str(config['solver']['passes']), '--top-m', '1',
            '--search-jobs', str(config['solver']['jobs']),
            '--search-budget-ms', str(round(config['solver']['time_budget_s'] * 1000)),
            '--artifact-cache', str(Path(config['device']['cache_dir']).expanduser()),
            '--dump-cg', str(directory / 'selected.mlir')]
        for feature in ('global_la','nonpaged_la','paged_la','paged_la_splitk'):
            if feature in config['features']:
                command += ['--'+feature.replace('_','-'),str(config['features'][feature])]
        atomic_json(directory / 'command.json', command)
        snapshot = source_snapshot(ROOT, compiler)
        atomic_json(directory / 'source.json', snapshot)
        with gpu_lock():
            with (directory / 'build.log').open('w') as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            identity = generate(library, snapshot, executor=config['features']['forward_executor'])
        record = dict(batch=batch, library=str(library), export=str(export), bridge=str(bridge),
            artifact_id=identity['artifact_id'], selection='predicted; no performance measurement')
        plans.append(record)
        atomic_json(out / 'plans.json', plans)
        print(json.dumps(dict(event='dnn_build_complete', **record)), flush=True)
    return plans


def run(config, plans, out, inputs):
    from safetensors.torch import load_file, save_file
    from .check import NativeForward
    if inputs is None:
        raise ValueError('run requires --inputs with exported user-input names in a safetensors file')
    values = load_file(str(inputs), device='cuda')
    for record in plans:
        with NativeForward(record['library'], record['export'], record['bridge'], record['batch']) as native:
            missing = set(native.inputs) - values.keys()
            if missing:
                raise ValueError('missing exported user inputs: ' + ', '.join(sorted(missing)))
            outputs = native([values[name] for name in native.inputs])
            save_file(dict(zip(native.outputs, outputs)), str(out / f'B{record["batch"]}' / 'outputs.safetensors'))


def check(config, plans, out):
    from .check import main as check_model
    model = config['model']
    data = config.get('data', {}).get('path')
    if not data:
        raise ValueError('data.path is required for the official G-DNN correctness gate')
    for record in plans:
        arguments = ['--model', model['name'], '--library', record['library'],
            '--export', record['export'], '--bridge', record['bridge'], '--data', data,
            '--batch', str(record['batch']), '--out', str(out / f'B{record["batch"]}' / 'correctness.json')]
        for option in ('checkpoint', 'nafnet_weights'):
            if model.get(option):
                arguments += ['--' + option.replace('_', '-'), str(model[option])]
        if model.get('attention_mask'):
            arguments.append('--mask')
        check_model(arguments)


def report(plans, out):
    rows = []
    for plan in plans:
        path = out / f'B{plan["batch"]}' / 'correctness.json'
        result = json.loads(path.read_text()) if path.is_file() else None
        if result and result['artifact_id'] != plan['artifact_id']:
            raise ValueError('correctness receipt belongs to another artifact')
        rows.append(dict(plan, correctness='pending' if result is None else
            'passed' if result['passed'] else 'failed', metrics=result.get('metrics') if result else None))
    atomic_json(out / 'report.json', dict(scope='implementation and correctness; no performance testing', plans=rows))
    print(json.dumps(rows), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('export', 'build', 'run', 'check', 'report'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--tilemega', type=Path, default=ROOT / 'build-dm/tools/tilemega')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--inputs', type=Path)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    out = (args.run_dir or Path(config['output']['dir'])).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'config.json', config)
    if args.command == 'export':
        model_export(config, out)
        return
    if args.command == 'build':
        build(config, out, args.tilemega.resolve())
        return
    plans = json.loads((out / 'plans.json').read_text())
    if args.command in ('run', 'check'):
        with gpu_lock():
            run(config, plans, out, args.inputs) if args.command == 'run' else check(config, plans, out)
    if args.command == 'report':
        report(plans, out)


if __name__ == '__main__':
    main()
