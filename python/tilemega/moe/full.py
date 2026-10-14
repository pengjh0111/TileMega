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


from tilemega.serving.memory import memory_report


def export_plans(config, out, batches, capacity, prompt_len=64):
    import torch
    from tilemega.serving.export import export
    from tilemega.export_bridge import serialize
    result = {}
    for phase in ('decode', 'prefill'):
        directory = out/'exports'/phase
        export(config, phase, capacity, directory, prompt_len)
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
        '--prompt-len', str(args.prompt_len),
        '--past-range', f'{args.prompt_len}:{max(args.prompt_len,args.prompt_len+args.steps-2)}' if phase == 'decode' else '0:0',
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
    from tilemega.build.identity import snapshot, verify
    checkpoint = index_check(args.checkpoint)
    atomic_json(args.out/'checkpoint-index.json', checkpoint)
    bridges = export_plans(config, args.out, args.batch, args.capacity, args.prompt_len)
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
            source = directory/'source.json'
            inputs = snapshot(ROOT, source, args.compiler)
            with (directory/'build.log').open('w') as stream:
                if args.command == 'dry-build':
                    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                else:
                    with gpu_lock():
                        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                        verify(binary, expected_source=inputs)
            manifest_path = Path(str(binary)+'.plan.json')
            manifest = json.loads(manifest_path.read_text()); manifests.append(manifest)
            recipes = {buffer['name']: buffer['recipe'] for buffer in manifest['buffers'] if 'recipe' in buffer}
            extents = {buffer['name']: sum(int(buffer.get(key, 0))*bound for key, bound in (
                ('constant', 1), ('per_seq', manifest['seq']), ('per_past', manifest['past_hi']),
                ('per_total', manifest['seq']+manifest['past_hi']), ('per_batch', batch)))
                for buffer in manifest['buffers'] if 'recipe' in buffer}
            index_check(args.checkpoint, recipes, recipe_elements=extents)
            plans.append(dict(batch=batch, phase=phase, binary=str(binary),
                bridge=str(bridges[phase, batch]), manifest=str(manifest_path),
                manifest_sha256=file_sha(manifest_path), binary_sha256=file_sha(binary)))
            atomic_json(args.out/'plans.json', plans)
            print(json.dumps(dict(event='moe_plan_built', batch=batch, phase=phase, binary=str(binary))), flush=True)
        atomic_json(args.out/f'B{batch}'/'memory.json', memory_report(config, manifests, batch, args.capacity))
    return plans


def generation_tokens(engine, rows, steps, mode, vocab_size, capture_step=None):
    import torch
    if mode not in (1,2) or not (engine.prefill_lib.info.modes & mode and
                                 engine.decode_lib.info.modes & mode):
        raise ValueError('requested correctness mode is absent from one plan')
    stream = torch.cuda.current_stream()
    engine.state.kv_storage.zero_()
    engine.state.tokens.fill_(-1)
    prompt_len = rows.shape[1]
    engine.state.tokens[:, :prompt_len].copy_(rows.to(
        device=engine.state.tokens.device, dtype=torch.int32))
    engine.prefill.launch(0, mode, stream.cuda_stream)
    if capture_step is not None:
        stream.synchronize();capture_step(engine.prefill, 'prefill')
    for step in range(steps-1):
        engine.decode.launch(step, mode, stream.cuda_stream)
        if capture_step is not None:
            stream.synchronize();capture_step(engine.decode, 'decode')
    stream.synchronize()
    tokens=engine.state.tokens[:, prompt_len:prompt_len+steps].cpu()
    if not torch.all((tokens>=0) & (tokens<vocab_size)):
        raise AssertionError('generated tokens are unwritten or outside the vocabulary')
    return tokens.tolist()


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
    if args.prompt_ids:
        from tilemega.serving.workload import validate_prompt_ids
        values=json.loads(args.prompt_ids.read_text())
        validate_prompt_ids(values,max(args.batch),args.prompt_len)
        rows=torch.tensor(values[:max(args.batch)],dtype=torch.int64)
    else:
        rows = torch.randint(0, config['vocab_size'], (max(args.batch), args.prompt_len), generator=generator)
    atomic_json(args.out/'prompt_ids.json', rows.tolist())
    receipts = []
    for batch in args.batch:
        pair = {item['phase']: Path(item['binary']) for item in plans if item['batch'] == batch}
        identities = {phase: verify(binary)['artifact_id'] for phase, binary in pair.items()}
        lower = json.loads((args.out/f'B{batch}'/'memory.json').read_text())
        routing = []
        capture_step = None
        reference_steps = min(args.steps, getattr(args, 'hf_max_tokens', None) or args.steps)
        if getattr(args, 'routing_check', False):
            from tilemega.moe.routing_check import index_names, capture
            names = {item['phase']: index_names(json.loads(Path(item['bridge']).read_text()),
                config['num_hidden_layers']) for item in plans if item['batch'] == batch}
            def capture_step(plan, phase):
                if len(routing) < reference_steps:
                    routing.append(capture(plan, names[phase], batch,
                        args.prompt_len if phase == 'prefill' else 1, config['num_experts_per_tok']))
        with gpu_lock():
            available, _ = torch.cuda.mem_get_info()
            if lower['estimated_allocation_bytes'] > available:
                raise MemoryError('insufficient GPU memory for the manifest allocation estimate')
            # Both modes use one immutable weight allocation and one prompt.
            # Reset request state so the second launch cannot read prior tokens/KV.
            with ServingEngine(args.checkpoint, pair['prefill'], pair['decode'], batch,
                    prompt_len=args.prompt_len,
                    max_new_tokens=args.steps, mode='L1', prefill_mode='L1',
                    decode_loop=False, step_events=False) as engine:
                tokens=[generation_tokens(engine,rows[:batch],args.steps,mode,config['vocab_size'],
                    capture_step if mode == 1 else None)
                        for mode in (1,2)]
            # Closing native plans retains Python tensor bindings. Release them
            # and cached allocations before the HF subprocess needs memory.
            del engine
            torch.cuda.empty_cache()
            if tokens[0] != tokens[1]:
                raise AssertionError('C-2: same-binary L1/L2 generation tokens differ')
        generated = args.out/f'B{batch}'/'tokens.json'; atomic_json(generated, tokens[0])
        if routing:
            from tilemega.moe.routing_check import compare_hf
            atomic_json(args.out/f'B{batch}'/'routing_indices.json', routing)
            with gpu_lock():
                compare_hf(args.checkpoint, rows[:batch].tolist(),
                           [row[:reference_steps] for row in tokens[0]], routing,
                           args.out/f'B{batch}'/'routing.json')
                torch.cuda.empty_cache()
        if args.hf_check:
            reference_tokens = generated
            if reference_steps < args.steps:
                reference_tokens = args.out/f'B{batch}'/'hf_tokens.json'
                atomic_json(reference_tokens, [row[:reference_steps] for row in tokens[0]])
            with gpu_lock():
                subprocess.run([sys.executable, '-m', 'tilemega.serving.hf_check',
                    '--model', str(args.checkpoint), '--prompt-ids', str(args.out/'prompt_ids.json'),
                    '--generated', str(reference_tokens), '--out', str(args.out/f'B{batch}'/'hf.json'),
                    '--experts-implementation', 'grouped_mm', '--skip-free-greedy'], check=True)
        receipts.append(dict(evidence='verified', batch=batch, steps=args.steps,
            same_binary_l1_l2_tokens=True, artifacts=identities, performance_measured=False,
            teacher_forced_steps=reference_steps if args.hf_check else 0,
            routing_steps=len(routing)))
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
    if any(report['exceeds_available'] for report in reports) and not getattr(args,'report_only',False):
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
    parser.add_argument('--prompt-len', type=int, default=64)
    parser.add_argument('--prompt-ids', type=Path)
    parser.add_argument('--report-only', action='store_true',
                        help='preflight records insufficient memory without allocating or failing')
    parser.add_argument('--pg', choices=('l2', 'pages'), default='l2')
    parser.add_argument('--hf-check', action='store_true')
    parser.add_argument('--hf-max-tokens', type=int,
                        help='limit HF and routing reference prefixes; C-2 still uses all steps')
    parser.add_argument('--routing-check', action='store_true', help='report teacher-forced routing agreement')
    args = parser.parse_args(argv)
    if not args.batch or len(set(args.batch)) != len(args.batch) or any(not 1<=b<=16 for b in args.batch):
        parser.error('batches must be distinct values in [1,16]')
    if args.steps<2 or args.prompt_len<1 or args.capacity<args.prompt_len+args.steps:
        parser.error('steps must be >=2 and capacity must cover the prompt plus output')
    if args.report_only and args.command!='preflight':
        parser.error('--report-only is only valid for preflight')
    if args.hf_max_tokens is not None and args.hf_max_tokens < 1:
        parser.error('--hf-max-tokens must be positive')
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
