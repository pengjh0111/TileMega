#!/usr/bin/env python3
"""Check compile's forward CG path without collecting timing data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from identity_dm import generate, sha, source_snapshot, verify


def tensor_sha(tensor):
    raw = tensor.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8)
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


def check(binary, out, batch, tokens):
    import torch
    from safetensors.torch import save_file
    from tilemega.serving.plan import FORWARD, PlanLibrary
    from tilemega.serving.weights import load_weights

    identity = verify(binary)
    library = PlanLibrary(binary)
    info = library.info
    assert info.phase == FORWARD and info.seq == (tokens or 1)
    assert (info.batch_lo, info.batch_hi) == (batch, batch)
    assert info.past_lo == info.past_hi == info.capacity == 0
    assert not any('kv' in buffer.name for buffer in library.buffers)
    torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = tokens or batch * 17
    x = torch.randn(rows, 128, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(64, 128, dtype=torch.bfloat16)
    checkpoint = out / 'checkpoint'
    checkpoint.mkdir(exist_ok=False)
    save_file({'weight': weight}, str(checkpoint / 'model.safetensors'))
    weights = load_weights(checkpoint, library)
    assert weights and set(weights) == {buffer.name for buffer in library.buffers
                                      if buffer.pack_json}
    actual = torch.empty(rows, 64, device='cuda', dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(x.float(), weight.cuda().float())
    pointers = {name: value.data_ptr() for name, value in weights.items()}
    pointers.update(x=x.data_ptr(), out=actual.data_ptr())
    records = []
    with library.create(batch, pointers, torch.cuda.current_device()) as plan:
        plan.set_steps([0])
        for invalid in ([1], [0, 0]):
            try:
                plan.set_steps(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('forward accepted a nonzero or repeated step')
        first = None
        for epoch in range(8):
            for mode in (1, 2):
                actual.fill_(float('nan'))
                plan.launch(0, mode, torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                error = (actual.float() - reference).abs()
                assert torch.all(error <= 1.6e-2 + 1.6e-2 * reference.abs())
                if first is None:
                    first = actual.clone()
                else:
                    assert torch.equal(first, actual)
                records.append(dict(epoch=epoch, mode=mode, passed=True,
                                    max_absolute_error=error.max().item()))
    assert verify(binary)['artifact_id'] == identity['artifact_id']
    result = dict(evidence='verified', passed=True, artifact_id=identity['artifact_id'],
        scope='forward CG through compile CLI, weight packing, ABI and numerical checks; '
              'no exported-model or fresh-process synchronization gate',
        batch=batch, tokens=tokens, rows=rows, input_sha256=tensor_sha(x),
        weight_sha256=tensor_sha(weight), output_sha256=tensor_sha(first),
        cases=records)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--batch', type=int, choices=(1, 8), default=1)
    parser.add_argument('--tokens', type=int, choices=(0, 1, 128, 4096), default=0)
    parser.add_argument('--pg', choices=('l2', 'pages'), default='l2')
    parser.add_argument('--weight-layout', choices=('row', 'tiled'), default='row')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.tokens and args.batch != 1:
        parser.error('token regions require batch one')
    if args.run:
        check(args.run, args.out, args.batch, args.tokens)
        return
    if args.target is None:
        parser.error('--target is required for a build')
    args.out.mkdir(parents=True, exist_ok=False)
    compiler = args.root / 'build-dm/tools/tilemega'
    source = source_snapshot(args.root, compiler)
    (args.out / 'source_before_build.json').write_text(json.dumps(source, indent=2) + '\n')
    cg = args.out / 'forward.mlir'
    subprocess.run([str(args.root / 'build-dm/tilemega-unit'), 'forward_frontend',
                    '--emit-token-cg' if args.tokens else '--emit-cg', str(cg),
                    str(args.tokens or 1)], check=True)
    binary = args.out / 'forward.so'
    command = [str(compiler), 'compile', str(cg), str(binary), '--emit', 'serving',
               '--serving', 'forward', '--batch', str(args.batch),
               '--runtime-target', str(args.target), '--pg', args.pg,
               '--weight-layout', args.weight_layout,
               '--nonpaged-weight-layout', args.weight_layout, '--pdl', 'off',
               '--sync', 'calibrated', '--dump-cg', str(args.out / 'lowered.mlir')]
    (args.out / 'compile_command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (args.out / 'compile.log').open('w') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    plan = json.loads(Path(str(binary) + '.plan.json').read_text())
    assert plan['phase'] == 'forward' and plan['seq'] == (args.tokens or 1)
    assert plan['capacity'] == plan['past_lo'] == plan['past_hi'] == 0
    identity = generate(binary, source, executor='L1/L2')
    assert identity.get('complete_macros'), 'forward CLI check requires complete macro capture'
    subprocess.run([sys.executable, __file__, '--root', str(args.root),
                    '--out', str(args.out), '--run', str(binary),
                    '--batch', str(args.batch), '--tokens', str(args.tokens)], check=True)
    if source['source_files'] != source_snapshot(args.root, compiler)['source_files']:
        raise RuntimeError('forward CLI build inputs changed during validation')
    print(json.dumps(dict(event='forward_cli_check_complete', passed=True,
                          artifact_id=identity['artifact_id'])), flush=True)


if __name__ == '__main__':
    main()
