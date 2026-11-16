#!/usr/bin/env python3
"""Source/asset release for parallel classic-model baseline collection."""
from __future__ import annotations

import json
from pathlib import Path

import dnn1_aitemplate as runner

WORKLOAD_ID = 'dnn-classic-v1'
MODELS = ('resnet18', 'resnet34', 'vgg16', 'alexnet', 'bert')
BATCHES = (1, 2, 4, 8, 16)
WORKLOAD = dict(id=WORKLOAD_ID, models=list(MODELS), batches=list(BATCHES))
CELLS = {(model, batch) for model in MODELS for batch in BATCHES}
RETAINED = {(model, batch) for model in ('resnet18', 'bert') for batch in BATCHES}
ENVIRONMENT = 'TILEMEGA_DNN1_AIT_RELEASE'


def read(path):
    return json.loads(Path(path).read_text())


def checked(record):
    path = Path(record['path'])
    if runner.sha(path) != record['sha256']:
        raise ValueError('classic release evidence changed: ' + str(path))
    return path


def check_workload(value):
    if value.get('workload_id') != WORKLOAD_ID or value.get('workload') != WORKLOAD:
        raise ValueError('the exact new classic25 workload is required')


def validate(path):
    path = Path(path).resolve()
    value = read(path)
    check_workload(value)
    if (value.get('schema') != 2 or value.get('system') != 'aitemplate' or
            value.get('parallel_baselines_authorized') is not True or
            value.get('gpu_dispatch_authorized') is not True or not value.get('authorization')):
        raise ValueError('source/assets must be explicitly released for parallel baselines')
    if value['source_identity'] != runner.source_identity():
        raise ValueError('classic AIT source differs from the CPU-frozen identity')
    checked(value['cpu_prepare'])
    checked(value['asset_preparation'])
    checked(value['retained_AIT'])
    assets = value['assets']
    if {(row['model'], row['batch']) for row in assets} != CELLS or len(assets) != 25:
        raise ValueError('classic release requires exactly25 immutable asset identities')
    if value['protocol_sources'] != protocol_sources():
        raise ValueError('classic collection protocol changed')
    fingerprints = dict(value['protocol_sources'])
    for row in assets:
        if not row.get('inputs_sha256'):
            raise ValueError('classic asset input-value digest is absent')
        for name in ('exported_program', 'checkpoint', 'manifest', 'inputs'):
            evidence = row['files'][name]
            fingerprints[str(Path(evidence['path']).resolve())] = evidence['sha256']
    for source, digest in fingerprints.items():
        checked(dict(path=source, sha256=digest))
    return dict(path=str(path), sha256=runner.sha(path), schema=2,
                workload_id=WORKLOAD_ID, workload=WORKLOAD,
                policy='parallel baselines; fixed source/assets; no dependency on TVM completion')


def protocol_sources():
    protocol_names = ('dnn1_aitemplate_cell.py', 'dnn1_aitemplate_collect.py',
        'dnn1_aitemplate_classic_release.py', 'dnn1_baselines.py', 'dnn1_device.py',
        'dnn1_baseline_boundaries.py', 'dnn1_background_cards.py',
        'dnn1_dispatch.py', 'dnn1_owned_controller.py', 'dnn1_guard_entry.py',
        'dnn1_fifo_background_python.py', 'dnn1_settling_background_python.py',
        'dnn1_preflight_settle.py', 'dnn1_preflight_settle_guard.patch',
        'dnn1_idle_floor_python.py', 'dnn1_resource_entry.py',
        'run_r16_with_device_identity.py')
    sources = {str(runner.ROOT / 'scripts' / name):
               runner.sha(runner.ROOT / 'scripts' / name) for name in protocol_names}
    for name in ('python/tilemega/bench/gpu_guard.py', 'python/tilemega/bench/scheduler.py',
                 'configs/dnn/classic_h800_devices.json'):
        source = runner.ROOT / name
        sources[str(source)] = runner.sha(source)
    import dnn1_baselines as protocol
    sources[str(protocol.POLICY)] = runner.sha(protocol.POLICY)
    allowlist = Path(read(protocol.POLICY)['gpu_allowlist_file'])
    sources[str(allowlist)] = runner.sha(allowlist)
    return sources


def create(config, destination):
    import csv
    check_workload(config)
    if (config.get('parallel_baselines_authorized') is not True or
            config.get('gpu_dispatch_authorized') is not True or not config.get('authorization')):
        raise ValueError('parallel classic baseline authorization is required')
    cpu_path = Path(config['cpu_prepare']).resolve()
    checked(dict(path=str(cpu_path), sha256=config['cpu_prepare_sha256']))
    cpu = read(cpu_path)
    check_workload(cpu)
    rows = cpu['rows']
    if (len(rows) != 25 or {(row['model'], row['batch']) for row in rows} != CELLS or
            any(row['status'] != 'ir_created' or row['precision'] != 'bf16' for row in rows) or
            cpu['identity'] != runner.source_identity() or cpu.get('input_seed') != 20261007):
        raise ValueError('classic25 CPU source/precision/input proof is incomplete or changed')
    assets = []
    for row in rows:
        if not row.get('outputs') or any(out['dtype'] != 'bfloat16' for out in row['outputs']):
            raise ValueError('classic primary outputs must remain BF16')
        files = row.get('asset_files', {})
        if set(files) != {'exported_program', 'checkpoint', 'manifest', 'inputs'}:
            raise ValueError('CPU proof must bind all original model assets')
        for evidence in files.values():
            checked(evidence)
        assets.append(dict(model=row['model'], batch=row['batch'], files=files,
                           inputs_sha256=row['inputs_sha256']))
    retained = config['retained_AIT']
    with checked(retained).open(newline='') as stream:
        previous = list(csv.DictReader(stream))
    if (len(previous) != 10 or {(r['model'], int(r['batch'])) for r in previous} != RETAINED or
            any(r.get('system') != 'aitemplate' or r['status'] != 'pass' or r['precision'] != 'bf16' or
                r['correctness_passes'] != '5/5' for r in previous)):
        raise ValueError('only the original ten accepted RN18/BERT cells may be reused')
    by_cell = {(r['model'], r['batch']): r for r in assets}
    for row in previous:
        asset = by_cell[row['model'], int(row['batch'])]
        if (row['exported_program_sha256'] != asset['files']['exported_program']['sha256'] or
                row['source_checkpoint_sha256'] != asset['files']['checkpoint']['sha256'] or
                row['inputs_sha256'] != asset['inputs_sha256']):
            raise ValueError('retained RN18/BERT data use different assets')
    asset_preparation = config['asset_preparation']
    preparation = read(checked(asset_preparation))
    if preparation.get('active_models') != list(MODELS):
        raise ValueError('asset preparation belongs to another workload')
    prepared = preparation.get('cells', [])
    if (len(prepared) != 15 or
            {(row['model'], row['batch']) for row in prepared} != CELLS - RETAINED):
        raise ValueError('asset preparation must bind exactly the fifteen new cells')
    for row in prepared:
        asset = by_cell[row['model'], row['batch']]
        expected = dict(exported_program_sha256=asset['files']['exported_program']['sha256'],
            source_checkpoint_sha256=asset['files']['checkpoint']['sha256'],
            manifest_sha256=asset['files']['manifest']['sha256'],
            inputs_file_sha256=asset['files']['inputs']['sha256'],
            expected_inputs_sha256=asset['inputs_sha256'])
        if row.get('status') != 'ready' or any(row.get(key) != digest for key, digest in expected.items()):
            raise ValueError('CPU import differs from the prepared new-model assets')
    sources = protocol_sources()
    value = dict(schema=2, system='aitemplate', workload_id=WORKLOAD_ID, workload=WORKLOAD,
        authorization=config['authorization'], parallel_baselines_authorized=True,
        gpu_dispatch_authorized=True,
        source_identity=cpu['identity'], protocol_sources=sources, assets=assets,
        retained_AIT=retained, asset_preparation=asset_preparation,
        cpu_prepare=dict(path=str(cpu_path), sha256=runner.sha(cpu_path)),
        reference='same stored BF16 checkpoint/input values promoted toFP32',
        precision='bf16', graph_mode=True, warmup=50, iterations=200,
        historical_TVM25_is_not_a_release_dependency=True)
    path = Path(destination) / 'classic_gpu_release.json'
    if path.exists():
        raise FileExistsError('retain the existing source/asset release; use a new epoch')
    runner.save(path, value)
    return validate(path)


def sample_assets(publication, model, batch, exported_program_sha256,
                  checkpoint_sha256, inputs_sha256):
    release = read(publication['path'])
    expected = next((row for row in release['assets']
                     if row['model'] == model and row['batch'] == batch), None)
    if (expected is None or expected['files']['exported_program']['sha256'] != exported_program_sha256 or
            expected['files']['checkpoint']['sha256'] != checkpoint_sha256 or
            expected['inputs_sha256'] != inputs_sha256):
        raise ValueError('sample differs from the source-frozen checkpoint/export/input values')
    return expected
