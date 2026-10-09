#!/usr/bin/env python3
"""Check the actual CG-generated layout/conv/LN shared library, without timing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def execute(root):
    import torch
    from tilemega.serving.plan import FORWARD, PlanLibrary
    torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32 = False
    library = PlanLibrary(root / 'generated-sm_89.so')
    pool = 'Run<TaskKind::kPool,' in (root/'generated.cu').read_text()
    assert library.info.phase == FORWARD and library.info.capacity == 0
    x = torch.randn(2, 3, 7, 11, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(19, 3, 3, 3, device='cuda', dtype=torch.bfloat16)
    packed = torch.zeros(19, 3, 3, 8, device='cuda', dtype=torch.bfloat16)
    packed[..., :3] = weight.permute(0, 2, 3, 1)
    gamma = torch.randn(19, device='cuda', dtype=torch.bfloat16)
    beta = torch.randn(19, device='cuda', dtype=torch.bfloat16)
    output = torch.empty(2, 2 if pool else 4, 3 if pool else 6, 19, device='cuda', dtype=torch.bfloat16)
    conv = torch.nn.functional.conv2d(x.float(), weight.float(), stride=2, padding=1)
    rounded = conv.to(torch.bfloat16).float()
    if pool:
        rounded = torch.nn.functional.max_pool2d(rounded, 3, 2, 1)
    rounded = rounded.permute(0, 2, 3, 1)
    reference = torch.nn.functional.layer_norm(rounded, [19], gamma.float(), beta.float(), 1e-6)
    buffers = dict(input=x, buffer2=packed, buffer4=gamma, buffer5=beta, output=output)
    assert {b.name for b in library.buffers if b.role == 1} == set(buffers)
    cases = []
    with library.create(2, {name: value.data_ptr() for name, value in buffers.items()}, 0) as plan:
        plan.set_steps([0])
        first = None
        for epoch in range(3):
            for mode in (1, 2):
                output.fill_(float('nan'))
                plan.launch(0, mode, torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                error = (output.float() - reference).abs()
                assert torch.all(error <= 1.6e-2 + 1.6e-2 * reference.abs()), error.max().item()
                if first is None:
                    first = output.clone()
                else:
                    assert torch.equal(first, output)
                cases.append(dict(epoch=epoch, mode=mode, max_error=error.max().item()))
    print(json.dumps(dict(event='generated_dnn_correctness', passed=True,
        scope='CG-generated layout/conv/pool/LN ABI; no full model gate', pool=pool, cases=cases)), flush=True)


def build(root, arch):
    sys.path.insert(0, str(root / 'framework'))
    from capture_macros_dm import capture
    from identity_dm import resources, sha
    preparation = json.loads((root / 'preparation.json').read_text())
    def unchanged():
        for path, digest in preparation['inputs'].items():
            assert sha(path) == digest, path
    unchanged()
    source = root / 'generated.cu'
    binary = root / f'generated-sm_{arch}.so'
    support = preparation['support']
    command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2', '-lineinfo',
        '--expt-relaxed-constexpr', '-UNDEBUG', f'-arch=sm_{arch}',
        f'-DTILEMEGA_ARCH_ID={arch*10}', '-DTILEMEGA_MIDPOINT_REFINE=0',
        '-shared', '-Xcompiler=-fPIC', '-cudart', 'shared', '-Xptxas=-v,-warn-spills',
        '-x', 'cu', '-I'+str(root/'include'),
        '-I'+str(root/'third_party/cutlass/include'),
        '-I'+str(root/'third_party/cutlass/tools/util/include'),
        str(source), *(str(root / name) for name in support), '-o', str(binary)]
    log = root / f'generated-sm_{arch}.build.log'
    with log.open('w') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    macros = root / f'generated-sm_{arch}.macros'
    capture(command, macros)
    unchanged()
    implementations = ['LayoutConvertTaskBody', 'ServingGemmTaskBody::RunDm', 'LayerNormTaskBody']
    if 'Run<TaskKind::kPool,' in source.read_text():
        implementations.append('PoolTaskBody')
    identity = dict(schema='tilemega.dm1.native-test.identity.v1', evidence='verified',
        source=preparation, scope='CG-generated DNN primitives', cu_sha256=sha(source),
        binary_sha256=sha(binary), command=command, target_arch=f'sm_{arch}',
        compiler=dict(path=command[0], sha256=sha(command[0]),
            version=subprocess.check_output([command[0], '--version'], text=True)),
        complete_macros=dict(path=str(macros/'capture.json'), sha256=sha(macros/'capture.json')),
        implementations=implementations,
        resources=resources(log.read_text()), execution=dict(phase='forward', modes=['L1', 'L2']))
    identity['artifact_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    Path(str(binary)+'.identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps(dict(event='generated_dnn_build', arch=arch, artifact_id=identity['artifact_id'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arch', type=int)
    args = parser.parse_args()
    if args.arch:
        build(args.root, args.arch)
    else:
        execute(args.root)
