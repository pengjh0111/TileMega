#!/usr/bin/env python3
"""CPU-only DNN3 assets: one ResNet50 export, its per-batch ONNX and the TVM proof.

`all` runs each stage in a bounded child with hidden CUDA and the reject shim.
VGG16 is linked from the retained dnn1 export and only re-verified, so its
exported program and every downstream digest stay those of dnn-classic-v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python'
EXTRA_PATH = '/root/TileMega/runs/r16/env/dnn-export/lib/python3.10/site-packages'
REJECT = ROOT / 'runs/dnn1/drafts/batch/review_compile_v4/reject_cuda.so'
RETAINED = Path('/root/TileMega-dnn1/runs/dnn1/exports')
VGG16_SHA256 = '566fffcb04bc75aed110066fe788b0a1055cf1233940ce067744f4041db20eae'
SEED = 20261007
BATCHES = (1, 2, 4, 8, 16)
MAX_GIB = 48


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def require_hidden_cuda():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'reject_cuda.so' not in os.environ.get('LD_PRELOAD', ''):
        raise RuntimeError('explicit hidden CUDA and reject shim required')
    import torch
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch._C._cuda_getDeviceCount = lambda: 0
    def forbidden(*_args, **_kwargs):
        raise RuntimeError('CUDA initialization forbidden for DNN3 CPU assets')
    torch.cuda._lazy_init = forbidden
    torch.cuda.init = forbidden
    return torch


def prepare_model(name, exports, stage):
    """Export once, then prove BF16 bits and the FP32 oracle against the eager module."""
    torch = require_hidden_cuda()
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.dnn import baselines, export
    from safetensors.torch import load_file, save_file
    import copy
    directory = exports / name
    if directory.exists():
        raise FileExistsError('preserve the existing asset: ' + str(directory))
    stage.mkdir(parents=True, exist_ok=False)
    digest_output = lambda value: baselines.inputs_digest(baselines.leaves(value), {})[0]
    hashes = {str(p): baselines.sha(p) for p in (Path(export.__file__), Path(baselines.__file__), Path(__file__))}
    record = dict(model=name, status='preparing', started=time.time(), pid=os.getpid(),
                  source_sha256=hashes, seed=SEED, CUDA_VISIBLE_DEVICES='', reject_cuda=True,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), CPU_threads=4, GPU_executed=False,
                  cells=[], validation=[])
    save(stage / 'result.json', record)
    original, created, pre = export.model, {}, {}
    def capture(*values, **keywords):
        module, upstream = original(*values, **keywords)
        created.update(module=module, upstream=upstream)
        # Pre-export outputs come from the very instance being exported.
        with torch.inference_mode():
            for batch in (1, 16):
                image = torch.randn((batch, 3, 224, 224),
                    generator=torch.Generator(device='cpu').manual_seed(SEED)).bfloat16()
                pre[batch] = dict(output=module(image).detach().clone(),
                                  input_digest=baselines.inputs_digest((image,), {})[0])
        return module, upstream
    export.model = capture
    try:
        export.export(name, exports, stage / 'fixtures', random_weights=True, seed=SEED)
    finally:
        export.model = original
    eager = created['module']
    program, loaded = baselines.load_export(directory)
    checkpoint = directory / 'checkpoint/model.safetensors'
    state = load_file(str(checkpoint))
    assert state.keys() == program.state_dict.keys() == eager.state_dict().keys()
    for key, value in state.items():
        assert torch.equal(value, program.state_dict[key]) and torch.equal(value, eager.state_dict()[key])
        if value.is_floating_point():
            assert value.dtype == torch.bfloat16
    record['checkpoint_state_tensors_exact'] = len(state)
    del state
    fp32_eager = copy.deepcopy(eager).float().eval()
    fp32_ep = baselines.portable_module(copy.deepcopy(program.module()), dtype=torch.float32)
    native = [spec.arg.name for spec in program.graph_signature.input_specs if spec.kind.name == 'USER_INPUT']
    assert native == ['x']
    for batch in BATCHES:
        inputs, kwargs = baselines.make_inputs(program, batch, SEED)
        inputs_digest, tensors = baselines.inputs_digest(inputs, kwargs)
        path = directory / f'inputs_b{batch}.safetensors'
        assert not path.exists()
        save_file(dict(zip(native, inputs)), str(path))
        with torch.inference_mode():
            actual = baselines.leaves(program.module()(*inputs, **kwargs))[0]
            expected = pre[batch]['output'] if batch in pre else eager(*inputs, **kwargs)
            if batch in pre:
                assert pre[batch]['input_digest'] == inputs_digest
            assert actual.dtype == expected.dtype == torch.bfloat16
            assert tuple(actual.shape) == tuple(expected.shape) == (batch, 1000)
            assert torch.isfinite(actual).all() and torch.equal(actual, expected)
            ref_inputs, ref_kwargs = baselines.convert_inputs(inputs, kwargs, dtype=torch.float32)
            reference = baselines.leaves(fp32_eager(*ref_inputs, **ref_kwargs))[0]
            ep_reference = baselines.leaves(fp32_ep(*ref_inputs, **ref_kwargs))[0]
            assert reference.dtype == ep_reference.dtype == torch.float32
            assert torch.isfinite(reference).all() and torch.equal(reference, ep_reference)
            metrics = baselines.numerical_metrics([actual], [reference], name)
            try:
                baselines.correctness_gate(name, metrics)
                gate = True
            except baselines.BaselineFailure:
                gate = False
        record['validation'].append(dict(batch=batch, bf16_EP_eager_bits_equal=True,
            eager_compared='pre-export' if batch in pre else 'same original module after export',
            output_shape=list(actual.shape), output_dtype=str(actual.dtype),
            output_sha256=digest_output(actual), FP32_forward_executed=True,
            FP32_EP_eager_bits_equal=True, FP32_reference_sha256=digest_output(reference),
            BF16_vs_same_weight_FP32=metrics, CPU_BF16_oracle_gate_passed=gate))
        record['cells'].append(dict(model=name, batch=batch, status='ready', export=str(directory),
            exported_program_sha256=loaded['artifacts']['exported_program.pt2'],
            source_checkpoint_sha256=baselines.sha(checkpoint),
            manifest_sha256=baselines.sha(directory / 'manifest.json'),
            inputs_file=str(path), inputs_file_sha256=baselines.sha(path),
            expected_inputs_sha256=inputs_digest, input_tensors=tensors))
        save(stage / 'result.json', record)
        print(json.dumps(dict(model=name, batch=batch, bf16_EP_eager_bits_equal=True, oracle_gate=gate)), flush=True)
    assert all(baselines.sha(p) == h for p, h in hashes.items())
    assert not torch.cuda.is_initialized()
    record.update(status='ready', ended=time.time(), torch_cuda_initialized=False,
        same_dynamic_asset_batches=list(BATCHES), dynamic_batch_range=[1, 64],
        asset=dict(exported_program=dict(path=str(directory / 'exported_program.pt2'),
                       sha256=baselines.sha(directory / 'exported_program.pt2')),
                   checkpoint=dict(path=str(checkpoint), sha256=baselines.sha(checkpoint)),
                   manifest=dict(path=str(directory / 'manifest.json'),
                       sha256=baselines.sha(directory / 'manifest.json'))),
        provenance=dict(constructor='torchvision.models.' + name, weights=None, pretrained_downloads=False,
            initialization_seed=SEED, input_seed=SEED, weight_dtype='bf16',
            reference='same stored BF16 checkpoint/input values promoted to FP32',
            architecture_forward_modified=False, versions=export.versions(),
            accuracy_eligible=False, timing_eligible=True),
        finite_FP32_validation_batches=list(BATCHES), unexecuted_FP32_batches=[])
    save(stage / 'result.json', record)


def onnx_entry(arguments):
    """Run the unchanged harness exporter inside this process's hidden CUDA."""
    import runpy
    require_hidden_cuda()
    harness = ROOT / 'python/tilemega/dnn/baselines.py'
    sys.argv = [str(harness), *arguments]
    runpy.run_path(str(harness), run_name='__main__')


def bounded(stage, name, arguments, timeout):
    """One attempt under the shared CPU build lock and address-space limit."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', LD_PRELOAD=str(REJECT), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join((str(ROOT / 'python'), EXTRA_PATH)),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command = [PYTHON, str(ROOT / 'scripts/run_sm90r1_build.py'), '--receipt', str(stage / f'{name}.bounded.json'),
               '--max-gib', str(MAX_GIB), '--timeout', str(timeout), '--lock', '/tmp/tilemega_dnn3_cpu_assets.lock',
               '--', PYTHON, str(Path(__file__).resolve()), *arguments]
    began = time.time()
    with (stage / f'{name}.log').open('w') as log:
        code = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT).returncode
    return dict(command=command, returncode=code, elapsed_s=time.time() - began, log=str(stage / f'{name}.log'))


def prepare_onnx(exports, stage, row):
    """Same recipe as classic onnx_prepare_v2: at most two attempts; input bytes stay frozen."""
    model, batch = row['model'], row['batch']
    inputs = Path(row['inputs_file'])
    record = dict(model=model, batch=batch, status='failed', attempts=[], CPU_only=True)
    for attempt in (1, 2):
        result = bounded(stage, f'onnx_{model}_b{batch}_attempt{attempt}',
            ['onnx-entry', '--system', 'export-onnx', '--export', row['export'],
             '--batch', str(batch), '--seed', str(SEED)], 900)
        record['attempts'].append(result)
        assert sha(inputs) == row['inputs_file_sha256'], 'export changed frozen input bytes'
        if result['returncode'] == 0:
            metadata = exports / model / f'onnx_b{batch}.json'
            value = json.loads(metadata.read_text())
            assert value['batch'] == batch and value['input_seed'] == SEED
            assert value['exported_program_sha256'] == row['exported_program_sha256']
            assert value['native_inputs_sha256'] == row['inputs_file_sha256']
            assert sha(value['path']) == value['sha256']
            record.update(status='pass', metadata=str(metadata), metadata_sha256=sha(metadata),
                          onnx_sha256=value['sha256'])
            break
    return record


def retained_vgg16(exports):
    """Link and re-verify the classic export; nothing below this path is regenerated."""
    link = exports / 'vgg16'
    if not link.is_symlink():
        link.symlink_to(RETAINED / 'vgg16')
    assert link.resolve() == RETAINED / 'vgg16'
    proof = json.loads((RETAINED / 'preparation_classic_v1.json').read_text())
    cells = [c for c in proof['cells'] if c['model'] == 'vgg16']
    assert [c['batch'] for c in cells] == list(BATCHES)
    directory = link.resolve()
    for cell in cells:
        assert cell['exported_program_sha256'] == VGG16_SHA256
        assert sha(directory / 'exported_program.pt2') == VGG16_SHA256
        assert sha(directory / 'checkpoint/model.safetensors') == cell['source_checkpoint_sha256']
        assert sha(directory / 'manifest.json') == cell['manifest_sha256']
        assert sha(cell['inputs_file']) == cell['inputs_file_sha256']
    rows = json.loads((RETAINED / 'preparation_classic_onnx_v2.json').read_text())
    rows = [r for r in rows if r['model'] == 'vgg16']
    assert [r['status'] for r in rows] == ['pass'] * 5
    for row in rows:
        value = json.loads(Path(row['metadata']).read_text())
        assert sha(row['metadata']) == row['metadata_sha256'] and sha(value['path']) == value['sha256'] == row['onnx_sha256']
        assert value['exported_program_sha256'] == VGG16_SHA256
    return cells, rows, proof


def prepare_all(exports):
    exports.mkdir(parents=True, exist_ok=True)
    stage = exports / 'preparation_dnn3_v1'
    proof_path = exports / 'preparation_dnn3_v1.json'
    onnx_path = exports / 'preparation_dnn3_onnx_v1.json'
    if proof_path.exists() or onnx_path.exists():
        raise FileExistsError('preserve the existing preparation receipts')
    stage.mkdir(parents=True, exist_ok=True)
    vgg16, vgg16_onnx, classic = retained_vgg16(exports)
    model_stage = stage / 'resnet50'
    run = bounded(stage, 'model_resnet50',
        ['model', '--model', 'resnet50', '--exports', str(exports), '--stage', str(model_stage)], 1800)
    if run['returncode']:
        raise SystemExit(f"ResNet50 export failed; retained log: {run['log']}")
    result = json.loads((model_stage / 'result.json').read_text())
    assert result['status'] == 'ready' and len(result['cells']) == 5
    # Image inputs depend only on seed and shape, so the two models must agree.
    assert ({c['batch']: c['expected_inputs_sha256'] for c in result['cells']} ==
            {c['batch']: c['expected_inputs_sha256'] for c in vgg16})
    rows = []
    for cell in result['cells']:
        rows.append(prepare_onnx(exports, stage, cell))
        save(onnx_path, rows + vgg16_onnx)
        print(json.dumps(dict(model=cell['model'], batch=cell['batch'], onnx=rows[-1]['status'])), flush=True)
        if rows[0]['status'] != 'pass':
            raise SystemExit('initial ONNX canary failed; do not spread a generic entry failure')
    record = dict(schema=1, status='ready', models=['resnet50', 'vgg16'], workload_id='dnn3-v1', seed=SEED,
        cells=result['cells'] + vgg16, prepared_cells=10, ended=time.time(),
        identity_source='preparation_dnn3_v1 CPU runtime-input proof',
        onnx_preparation=str(onnx_path), onnx_status={f"{r['model']}_b{r['batch']}": r['status'] for r in rows + vgg16_onnx},
        assets=dict(resnet50=dict(result=str(model_stage / 'result.json'), result_sha256=sha(model_stage / 'result.json'),
            **result['asset'], validation=result['validation'], provenance=result['provenance'], command=run)),
        retained=dict(vgg16=dict(source=str(RETAINED / 'preparation_classic_v1.json'),
            source_sha256=sha(RETAINED / 'preparation_classic_v1.json'), exported_program_sha256=VGG16_SHA256,
            regenerated=False, provenance=classic['assets']['vgg16']['provenance'])),
        provenance=dict(CPU_only=True, pretrained_downloads=False, weights=None, precision='bf16',
            generator=str(Path(__file__).resolve()), generator_sha256=sha(__file__),
            performance_or_GPU_support_claim=False))
    if any(r['status'] != 'pass' for r in rows):
        record['status'] = 'needs_diagnosis'
    save(proof_path, record)
    print(json.dumps(dict(status=record['status'], proof=str(proof_path), sha256=sha(proof_path))), flush=True)
    return 0 if record['status'] == 'ready' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('all')
    p.add_argument('--exports', type=Path, default=ROOT / 'runs/dnn3/exports')
    p = sub.add_parser('model')
    p.add_argument('--model', required=True)
    p.add_argument('--exports', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    sub.add_parser('onnx-entry', add_help=False)
    options, rest = parser.parse_known_args()
    if options.action == 'onnx-entry':
        return onnx_entry(rest)
    if rest:
        parser.error('unexpected arguments: ' + ' '.join(rest))
    if options.action == 'model':
        return prepare_model(options.model, options.exports.resolve(), options.stage.resolve())
    return prepare_all(options.exports.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
