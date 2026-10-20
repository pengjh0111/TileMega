"""Prepare one fixed random AS-1 graph and compare a native output file."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tilemega.export_bridge import serialize

ROOT = Path('/home/d50064638/junhuip/TileMega/runs/as1')
SEED = 20261016


def controlled(path: Path) -> Path:
    path = path.absolute()
    if not path.is_relative_to(ROOT) or path == ROOT or path.resolve() != path:
        raise ValueError('Artifact path must be canonical and below runs/as1')
    return path


def write_tensor(path, tensor):
    tensor = tensor.detach().cpu().contiguous()
    data = tensor.view(torch.uint8).numpy().tobytes()
    path.write_bytes(data)
    return dict(file=path.name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read_tensor(path, shape, dtype):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).clone().reshape(shape)


def prepare_e0(out: Path, shape_index: int):
    out = controlled(out)
    if out.exists():
        raise FileExistsError('Use a new artifact directory')
    m, n, k = ((64, 128, 256), (33, 96, 129))[shape_index]
    torch.manual_seed(SEED)
    model = torch.nn.Linear(k, n, bias=False, dtype=torch.float32).eval()
    with torch.no_grad():
        model.weight.copy_(torch.randn(n, k) * 0.125)
    model = model.to(torch.bfloat16)
    x = (torch.randn(m, k) * 0.25).to(torch.bfloat16)
    with torch.no_grad():
        fp32 = torch.nn.functional.linear(x.float(), model.weight.float())
        bf16 = model(x)
    # Export a dynamic row domain; runtime binding chooses the fixed workload M.
    program = torch.export.export(model, (x[:4],),
        dynamic_shapes=({0: torch.export.Dim('rows', min=1, max=128)},))
    bridge = serialize(program)
    out.mkdir(parents=True)
    (out/'export.json').write_text(json.dumps(bridge, indent=2)+'\n')
    tensors = {
        'input': write_tensor(out/'input.bin', x),
        'weight_nk': write_tensor(out/'weight_nk.bin', model.weight),
        'weight_kn': write_tensor(out/'weight_kn.bin', model.weight.T.contiguous()),
        'reference_fp32': write_tensor(out/'reference_fp32.bin', fp32),
        'reference_bf16': write_tensor(out/'reference_bf16.bin', bf16),
    }
    config = dict(test_id='E-0', seed=SEED, model='gemm', shape_mnk=[m,n,k],
                  dtype='bf16', accumulator='fp32', tensors=tensors,
                  bridge='export.json', pipeline_status='prepared_not_executed',
                  weight_layout='logical N,K; native implementation packs K,N',
                  tolerance=dict(atol=0.001, rtol=0.0078125,
                      reason='one BF16 relative spacing, with 1e-3 near-zero allowance; FP32 accumulation'))
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(dict(status='prepared', out=str(out), shape_mnk=[m,n,k], seed=SEED)))


def error_stats(actual, reference, atol, rtol):
    a, r = actual.float(), reference.float()
    diff = (a-r).abs()
    finite = torch.isfinite(a) & torch.isfinite(r)
    bad = ~finite | (diff > atol + rtol*r.abs())
    return dict(max_abs_error=float(diff.max()),
                max_rel_error=float((diff/r.abs().clamp_min(1e-30)).max()),
                rms_error=float(diff.square().mean().sqrt()),
                nonfinite_elements=int((~finite).sum()), failed_elements=int(bad.sum()),
                elements=a.numel(), numerical_pass=not bool(bad.any()))


def compare(directory: Path, output: Path, pipeline: Path | None = None):
    def require(condition, message):
        if not condition:
            raise RuntimeError(message)
    directory, output = controlled(directory), controlled(output)
    cfg = json.loads((directory/'config.json').read_text())
    m,n,_ = cfg['shape_mnk']; shape=[m,n]
    actual = read_tensor(output, shape, torch.bfloat16)
    refs = {name: read_tensor(directory/cfg['tensors'][name]['file'], shape, dtype)
            for name,dtype in [('reference_fp32',torch.float32),('reference_bf16',torch.bfloat16)]}
    tol=cfg['tolerance']
    stats={name:error_stats(actual,ref,tol['atol'],tol['rtol']) for name,ref in refs.items()}
    result=dict(test_id=cfg['test_id'], seed=cfg['seed'], shape=shape, dtype='bf16',
                tolerance=tol, references=stats, numerical_pass=all(s['numerical_pass'] for s in stats.values()),
                output_file=str(output), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                c0_status='not_evaluated_body_probe', c2_status='not_met_single_executor',
                c3_status='pending_exact_executor_repeats')
    if pipeline is not None:
        pipeline=controlled(pipeline)
        generated=json.loads((pipeline/'pipeline.json').read_text())
        plan=json.loads((pipeline/'plan.json').read_text())
        job=json.loads((output.parent/'job.json').read_text())
        records=[json.loads(line) for line in (output.parent/'stdout.log').read_text().splitlines()
                 if line.startswith('{')]
        launch=next(r for r in records if r.get('tilemega_pipeline'))
        trace=json.loads(Path(str(output)+'.trace.json').read_text())
        require(job['status']=='passed' and launch['host_kernel_launches']==1,'Launch receipt failed')
        require(launch['plan_trace_pass'] and launch['abi_roundtrip_pass'],'Plan/ABI receipt failed')
        require(generated['tasks']==launch['plan_tasks']==len(plan['tasks'])==len(trace),'Task count mismatch')
        for task,observed in zip(plan['tasks'],trace):
            require(task['logical_task']==observed['tile'] and task['owner']==observed['worker'],'Task ownership mismatch')
            require(observed['executions']==observed['event']==plan['epochs'],'Task event/epoch mismatch')
        require('tmcg.tile_space' in (pipeline/'ctg.mlir').read_text(),'Missing CTG evidence')
        require('tmexec.skeleton' in (pipeline/'solved.mlir').read_text(),'Missing Solver evidence')
        executable=Path(job['command'][0])
        require(hashlib.sha256(executable.read_bytes()).hexdigest()==job['executable_sha256'],'Selected binary changed')
        result.update(c0_status='passed', single_persistent_launch_verified=True,
            pipeline_directory=str(pipeline), physical_device_id=job['physical_device_id'],
            logical_device_id=launch['logical_device_id'], binary_sha256=job['executable_sha256'],
            codegen_sha256=hashlib.sha256((pipeline/'kernel.cpp').read_bytes()).hexdigest(),
            device_locked=False, plan_tasks=len(plan['tasks']), groups=plan['launch_groups'],
            epochs=plan['epochs'], other_processes_present=job['other_processes_present_before'])
    (output.parent/'numerical_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['numerical_pass'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare-e0');prepare.add_argument('--out',type=Path,required=True)
    prepare.add_argument('--shape-index',type=int,choices=[0,1],required=True)
    check=sub.add_parser('compare');check.add_argument('--directory',type=Path,required=True)
    check.add_argument('--output',type=Path,required=True)
    check.add_argument('--pipeline-directory',type=Path)
    args=p.parse_args()
    if args.command=='prepare-e0':
        prepare_e0(args.out,args.shape_index);return 0
    return compare(args.directory,args.output,args.pipeline_directory)


if __name__=='__main__':
    raise SystemExit(main())
