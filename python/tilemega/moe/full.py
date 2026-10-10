"""Build full-depth MoE plans and check generation without collecting timings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tilemega.cache import atomic_json, file_sha
from tilemega.fingerprint import ROOT


def resolve_target(args):
    if str(args.target) != 'auto':
        return Path(args.target).resolve()
    from tilemega.dnn.cli import gpu_lock
    with gpu_lock():
        probe = subprocess.run([str(args.compiler), 'probe', 'device'],
            cwd=ROOT, capture_output=True, text=True, check=True)
    device = json.loads(probe.stdout)
    arch = device['arch_tag']
    if arch not in ('sm_80', 'sm_89', 'sm_90', 'sm_100', 'sm_120'):
        raise ValueError('unsupported full-model device architecture '+arch)
    profile = ROOT/'configs'/'targets'/f'{arch}.json'
    target = json.loads(profile.read_text())
    for key in ('arch_tag', 'sm_major', 'sm_minor', 'caps'):
        target[key] = device[key]
    target['resources'].update(device['resources'])
    # Resource discovery is not calibration. Retain the profile's provenance.
    target['dm_device_binding'] = dict(evidence='verified', device=device,
        profile=str(profile), profile_sha256=file_sha(profile),
        calibration_scope='existing architecture profile; no new measurements')
    path = args.out/'target.json'
    atomic_json(path, target)
    return path


def memory_report(config, manifests, batch, capacity):
    """Account for shared recipes, request state and both plans' buffers.

    CUDA context, event/queue tables and runtime-inserted split-K storage
    are not manifest buffers. Content-equal weights can also be deduplicated
    by the loader. This estimate cannot certify that a plan fits.
    """
    widths = {'bf16': 2, 'f32': 4, 'i32': 4, 'i64': 8}
    weights, names, phases = {}, {}, []
    for manifest in manifests:
        if manifest['batch_lo'] != batch or manifest['batch_hi'] != batch or \
                manifest['capacity'] != capacity:
            raise ValueError('memory report workload differs from the plan')
        seq = manifest['seq']; past = manifest['past_hi']
        internal = int(manifest.get('memory_arena_bytes', 0))
        for buffer in manifest['buffers']:
            elements = sum(int(buffer.get(k, 0))*v for k, v in (
                ('constant', 1), ('per_seq', seq), ('per_past', past),
                ('per_total', seq+past), ('per_batch', batch)))
            size = elements*widths[buffer['dtype']]
            if 'recipe' in buffer:
                recipe = json.dumps(buffer['recipe'], sort_keys=True, separators=(',', ':'))
                key = (recipe, buffer['dtype'], size)
                prior = names.setdefault(buffer['name'], key)
                if prior != key:
                    raise ValueError('phase plans disagree on weight '+buffer['name'])
                weights[key] = size
            elif buffer['role'] == 'internal' and 'dm_arena_offset' not in buffer:
                internal += size
        phases.append(dict(phase=manifest['phase'], internal_bytes=internal))
    layers = int(config['num_hidden_layers']); heads = int(config['num_key_value_heads'])
    dim = int(config.get('head_dim', config['hidden_size']//config['num_attention_heads']))
    state = layers*2*batch*heads*capacity*dim*2 + batch*capacity*4 + 2*capacity*dim*2
    return dict(evidence='inferred', scope='manifest estimate; excludes CUDA/event tables and inserted split-K storage; recipe-based weight deduplication',
        batch=batch, shared_weight_bytes=sum(weights.values()), request_state_bytes=state,
        phases=phases, estimated_allocation_bytes=sum(weights.values())+state+
            sum(phase['internal_bytes'] for phase in phases))


def export_plans(config, out, batches, capacity):
    import torch
    from tilemega.serving.export import export
    from tilemega.export_bridge import serialize
    result = {}
    for phase in ('decode', 'prefill'):
        directory = out/'exports'/phase
        export(config, phase, capacity, directory)
        program = torch.export.load(directory/'exported_program.pt2')
        for batch in batches:
            inputs = set(program.graph_signature.user_inputs)
            ids = next(node for node in program.graph.nodes
                       if node.name in inputs and node.name == 'input_ids')
            symbols = {str(ids.meta['val'].shape[0]): batch}
            bridge = directory/f'B{batch}.json'
            atomic_json(bridge, serialize(program, shape_bindings=symbols))
            result[phase, batch] = bridge
    return result


def compile_command(args, bridge, binary, phase, batch):
    return [str(args.compiler), 'compile', str(bridge), str(binary),
        '--frontend', 'decoder', '--emit', 'serving', '--serving', phase,
        '--batch', str(batch), '--capacity', str(args.capacity), '--pg', args.pg,
        '--past-range', f'1:{args.capacity-1}' if phase == 'decode' else '0:0',
        '--nonpaged-weight-layout', 'tiled', '--runtime-target', str(args.target)]


def deployment_layout_decision(config, decode_manifest, batch, capacity, target):
    report=memory_report(config, [decode_manifest], batch, capacity)
    available=int(target.get('resources', {}).get('dram_capacity_bytes', 0))
    # The opposite phase has not been built yet. Its workspace remains an
    # estimate; the later two-manifest report/preflight still checks it.
    estimate=2*report['shared_weight_bytes']+report['request_state_bytes']+ \
        2*report['phases'][0]['internal_bytes']
    return dict(evidence='inferred', device_capacity_bytes=available,
        two_layout_estimate_bytes=estimate,
        require_shared_layout=bool(available and estimate>available),
        scope='decode packed allocations and doubled workspace estimate; unknown capacity leaves the domain unrestricted')


def build_plans(args, config):
    from tilemega.moe.checkpoints import index_check
    from tilemega.dnn.cli import gpu_lock
    from tilemega.build.identity import source_snapshot, generate
    checkpoint = index_check(args.checkpoint)
    atomic_json(args.out/'checkpoint-index.json', checkpoint)
    bridges = export_plans(config, args.out, args.batch, args.capacity)
    plans = []
    target=json.loads(args.target.read_text())
    for batch in args.batch:
        manifests = []
        for phase in ('decode', 'prefill'):
            directory = args.out/f'B{batch}'/phase
            directory.mkdir(parents=True, exist_ok=True)
            binary = directory/('plan.cu' if args.command == 'dry-build' else 'plan.so')
            command = compile_command(args, bridges[phase, batch], binary, phase, batch)
            if phase == 'prefill':
                decision=deployment_layout_decision(config,manifests[0],batch,args.capacity,target)
                atomic_json(args.out/f'B{batch}'/'layout-decision.json',decision)
                if decision['require_shared_layout']:
                    reference=args.out/f'B{batch}'/'decode'/('plan.cu' if args.command=='dry-build' else 'plan.so')
                    command += ['--shared-weight-layout', str(reference)+'.plan.json']
            atomic_json(directory/'command.json', command)
            snapshot = source_snapshot(ROOT, args.compiler)
            atomic_json(directory/'source.json', snapshot)
            with (directory/'build.log').open('w') as stream:
                if args.command == 'dry-build':
                    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                else:
                    with gpu_lock():
                        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                        generate(binary, snapshot, executor='L2')
            manifest_path = Path(str(binary)+'.plan.json')
            manifest = json.loads(manifest_path.read_text()); manifests.append(manifest)
            recipes = {buffer['name']: buffer['recipe'] for buffer in manifest['buffers'] if 'recipe' in buffer}
            index_check(args.checkpoint, recipes)
            plans.append(dict(batch=batch, phase=phase, binary=str(binary),
                bridge=str(bridges[phase, batch]), manifest=str(manifest_path),
                manifest_sha256=file_sha(manifest_path), binary_sha256=file_sha(binary)))
            atomic_json(args.out/'plans.json', plans)
            print(json.dumps(dict(event='moe_plan_built', batch=batch, phase=phase, binary=str(binary))), flush=True)
        atomic_json(args.out/f'B{batch}'/'memory.json', memory_report(config, manifests, batch, args.capacity))
    return plans


def check_plans(args, config):
    import torch
    from tilemega.dnn.cli import gpu_lock
    from tilemega.serving.engine import ServingEngine
    from tilemega.build.identity import verify
    plans = json.loads((args.out/'plans.json').read_text())
    for item in plans:
        if file_sha(Path(item['manifest'])) != item['manifest_sha256'] or \
                file_sha(Path(item['binary'])) != item['binary_sha256']:
            raise ValueError('full-model plan artifact changed since construction')
    generator = torch.Generator().manual_seed(20261010)
    rows = torch.randint(0, config['vocab_size'], (max(args.batch), 64), generator=generator)
    atomic_json(args.out/'prompt_ids.json', rows.tolist())
    receipts = []
    for batch in args.batch:
        pair = {item['phase']: Path(item['binary']) for item in plans if item['batch'] == batch}
        identities = {phase: verify(binary)['artifact_id'] for phase, binary in pair.items()}
        lower = json.loads((args.out/f'B{batch}'/'memory.json').read_text())
        with gpu_lock():
            available, _ = torch.cuda.mem_get_info()
            if lower['estimated_allocation_bytes'] > available:
                raise MemoryError('insufficient GPU memory for the manifest allocation estimate')
            tokens = []
            for mode in ('L1', 'L2'):
                with ServingEngine(args.checkpoint, pair['prefill'], pair['decode'], batch,
                        max_new_tokens=args.steps, mode=mode, decode_loop=False,
                        step_events=False) as engine:
                    stream = torch.cuda.current_stream()
                    engine.state.tokens[:, :64].copy_(rows[:batch].to(device='cuda', dtype=torch.int32))
                    engine.prefill.launch(0, engine.prefill_mode, stream.cuda_stream)
                    for step in range(args.steps-1):
                        engine.decode.launch(step, engine.decode_mode, stream.cuda_stream)
                    stream.synchronize()
                    tokens.append(engine.state.tokens[:, 64:64+args.steps].cpu().tolist())
                del engine
            if tokens[0] != tokens[1]:
                raise AssertionError('C-2: same-binary L1/L2 generation tokens differ')
        generated = args.out/f'B{batch}'/'tokens.json'; atomic_json(generated, tokens[0])
        if args.hf_check:
            with gpu_lock():
                subprocess.run([sys.executable, '-m', 'tilemega.serving.hf_check',
                    '--model', str(args.checkpoint), '--prompt-ids', str(args.out/'prompt_ids.json'),
                    '--generated', str(generated), '--out', str(args.out/f'B{batch}'/'hf.json'),
                    '--experts-implementation', 'grouped_mm', '--skip-free-greedy'], check=True)
        receipts.append(dict(evidence='verified', batch=batch, steps=args.steps,
            same_binary_l1_l2_tokens=True, artifacts=identities, performance_measured=False))
        atomic_json(args.out/'correctness.json', receipts)
    return receipts


def preflight(args):
    import torch
    from tilemega.dnn.cli import gpu_lock
    reports = []
    with gpu_lock():
        available, capacity = torch.cuda.mem_get_info()
    for batch in args.batch:
        report = json.loads((args.out/f'B{batch}'/'memory.json').read_text())
        reports.append(dict(batch=batch,
            estimated_allocation_bytes=report['estimated_allocation_bytes'],
            available_bytes=available, device_capacity_bytes=capacity,
            exceeds_available=report['estimated_allocation_bytes'] > available))
    atomic_json(args.out/'memory-preflight.json', dict(evidence='inferred',
        scope='manifest estimate compared to available memory; not a fit certificate',
        batches=reports))
    if any(report['exceeds_available'] for report in reports):
        raise MemoryError('insufficient GPU memory for the manifest allocation estimate; '
                          'see '+str(args.out/'memory-preflight.json'))
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dry-build', 'preflight', 'build', 'check'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, default=ROOT/'build-dm/tools/tilemega')
    parser.add_argument('--target', default=str(ROOT/'configs/targets/sm_89.json'),
                        help='target JSON, or auto to bind the current device resources')
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--capacity', type=int, default=1088)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--pg', choices=('l2', 'pages'), default='l2')
    parser.add_argument('--hf-check', action='store_true')
    args = parser.parse_args(argv)
    if not args.batch or len(set(args.batch)) != len(args.batch) or any(not 1<=b<=16 for b in args.batch):
        parser.error('batches must be distinct values in [1,16]')
    if not 2<=args.steps<=1024 or args.capacity<64+args.steps:
        parser.error('steps must be in [2,1024] and capacity must cover the 64-token prompt')
    args.checkpoint=args.checkpoint.resolve();args.out=args.out.resolve()
    args.compiler=args.compiler.resolve()
    config = json.loads((args.checkpoint/'config.json').read_text())
    if config.get('model_type') != 'qwen3_moe':
        parser.error('checkpoint must be Qwen3 MoE')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'preflight':
        return preflight(args)
    if args.command != 'check':
        args.target = resolve_target(args)
    return check_plans(args, config) if args.command == 'check' else build_plans(args, config)


if __name__ == '__main__':
    main()
