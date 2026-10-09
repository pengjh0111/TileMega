#!/usr/bin/env python3
"""Check the actual CG-generated layout/conv/LN shared library, without timing."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys


def execute(root):
    sys.path.insert(0, str(root/'python'))
    import torch
    from tilemega.serving.plan import FORWARD, PlanLibrary
    torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32 = False
    library = PlanLibrary(root / 'generated-sm_89.so')
    if 'RunDepthwise<' in (root/'generated.cu').read_text():
        return execute_depthwise(torch, library, (root/'generated.cu').read_text())
    if 'Run<TaskKind::kGlobalPoolReduce,' in (root/'generated.cu').read_text():
        return execute_global(torch, library)
    if 'Run<TaskKind::kEncoderAttention,' in (root/'generated.cu').read_text():
        return execute_encoder(torch, library)
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


def execute_encoder(torch, library):
    from tilemega.serving.plan import FORWARD
    assert library.info.phase == FORWARD and library.info.capacity == 0 and library.info.seq == 128
    qkv = torch.randn(2, 128, 3, 3, 64, device='cuda', dtype=torch.bfloat16)
    output = torch.empty(2, 128, 192, device='cuda', dtype=torch.bfloat16)
    mask = torch.ones(2, 128, device='cuda', dtype=torch.int64)
    mask[0, 103:] = 0
    mask[1, :37] = 0
    q, k, v = (qkv[:, :, :, index].permute(0, 2, 1, 3).float() for index in range(3))
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v,
        attn_mask=mask[:, None, None, :].bool(), is_causal=False).permute(0, 2, 1, 3).reshape_as(output)
    buffers = dict(qkv=qkv, context=output, mask=mask)
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
    print(json.dumps(dict(event='generated_encoder_correctness', passed=True,
        scope='CG-generated masked encoder attention ABI against PyTorch SDPA; no full BERT gate', cases=cases)), flush=True)


def execute_global(torch, library):
    from tilemega.serving.plan import FORWARD
    assert library.info.phase == FORWARD and library.info.capacity == 0 and library.info.seq == 1
    x = torch.randn(2, 49, 19, device='cuda', dtype=torch.float32)
    partials = torch.full((2, 7, 19), float('nan'), device='cuda')
    for image in range(2):
        for tile in range(image*49//16, ((image+1)*49+15)//16):
            begin, end = max(tile*16, image*49)-image*49, min((tile+1)*16, (image+1)*49)-image*49
            partials[image, tile] = x[image, begin:end].sum(0)
    guarded = torch.full((2*19+32,), -12345., device='cuda')
    output = guarded[16:-16].view(2, 19)
    reference = x.double().mean(1)
    buffers = dict(partials=partials, mean=output)
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
                error = (output.double() - reference).abs()
                assert torch.all(error <= 1e-6 + 1e-6 * reference.abs()), error.max().item()
                assert torch.all(guarded[:16] == -12345.) and torch.all(guarded[-16:] == -12345.)
                if first is None:
                    first = output.clone()
                else:
                    assert torch.equal(first, output)
                cases.append(dict(epoch=epoch, mode=mode, max_error=error.max().item()))
    print(json.dumps(dict(event='generated_global_correctness', passed=True,
        scope='CG-generated segmented global pooling ABI against FP64 mean; no full CNN gate', cases=cases)), flush=True)


def execute_depthwise(torch, library, source):
    from tilemega.serving.plan import FORWARD
    assert library.info.phase == FORWARD and library.info.capacity == 0 and library.info.seq == 1
    row_band = int(re.search(r'RunDepthwise<(\d+),', source)[1])
    gated = row_band == 2
    channels, output_channels = (32, 16) if gated else (19, 19)
    cp, output_cp = (channels+7)//8*8, (output_channels+7)//8*8
    x = torch.randn(2, channels, 7, 11, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(channels, 1, 3, 3, device='cuda', dtype=torch.bfloat16)
    packed = torch.zeros(channels, 3, 3, 8, device='cuda', dtype=torch.bfloat16)
    packed[..., 0] = weight[:, 0]
    bias = torch.randn(channels, device='cuda', dtype=torch.float32)
    input_storage = torch.zeros(2, 9, 13, cp, device='cuda', dtype=torch.bfloat16)
    input_storage[:, 1:8, 1:12, :channels] = x.permute(0, 2, 3, 1)
    output_storage = torch.empty(2, 9, 13, output_cp, device='cuda', dtype=torch.bfloat16)
    partials = torch.empty(2, (7+row_band-1)//row_band, output_channels, device='cuda')
    dot = torch.nn.functional.conv2d(x.float(), weight.float(), bias, padding=1, groups=channels)
    if gated:
        reference = (dot[:, :output_channels] * dot[:, output_channels:]).to(torch.bfloat16)
    else:
        reference = dot.clamp(0, 6).to(torch.bfloat16)
    reference = reference.permute(0, 2, 3, 1)
    written = torch.zeros_like(output_storage, dtype=torch.bool)
    written[:, 1:8, 1:12, :output_channels] = True
    buffers = dict(input=input_storage, weight=packed, output=output_storage, bias=bias, partials=partials)
    mean = None
    if any(b.name == 'mean' for b in library.buffers):
        mean = torch.empty(2, output_channels, device='cuda')
        buffers['mean'] = mean
    assert {b.name for b in library.buffers if b.role == 1} == set(buffers)
    cases = []
    with library.create(2, {name: value.data_ptr() for name, value in buffers.items()}, 0) as plan:
        plan.set_steps([0])
        first = first_partials = None
        for epoch in range(3):
            for mode in (1, 2):
                output_storage.fill_(-12345)
                partials.fill_(float('nan'))
                if mean is not None:
                    mean.fill_(float('nan'))
                plan.launch(0, mode, torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                output = output_storage[:, 1:8, 1:12, :output_channels]
                error = (output.float() - reference.float()).abs()
                assert torch.all(error <= 1.6e-2 + 1.6e-2 * reference.float().abs()), error.max().item()
                assert torch.all(output_storage[~written] == torch.tensor(-12345., dtype=torch.bfloat16, device='cuda'))
                for band in range(partials.shape[1]):
                    expected = output[:, band*row_band:min((band+1)*row_band, 7)].double().sum((1, 2))
                    assert torch.all((partials[:, band].double()-expected).abs() <= 1e-5+1e-5*expected.abs())
                if mean is not None:
                    expected = output.double().mean((1, 2))
                    assert torch.all((mean.double()-expected).abs() <= 1e-6+1e-6*expected.abs())
                if first is None:
                    first, first_partials = output_storage.clone(), partials.clone()
                else:
                    assert torch.equal(first, output_storage) and torch.equal(first_partials, partials)
                cases.append(dict(epoch=epoch, mode=mode, max_error=error.max().item()))
    print(json.dumps(dict(event='generated_depthwise_correctness', passed=True,
        scope='CG-generated depthwise halo/chain/partial ABI; no full model gate', gated=gated, cases=cases)), flush=True)


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
    implementations = (['DepthwiseConvTaskBody'] if 'RunDepthwise<' in source.read_text()
        else ['GlobalPoolReduceTaskBody'] if 'Run<TaskKind::kGlobalPoolReduce,' in source.read_text()
        else ['EncoderAttentionTaskBody'] if 'Run<TaskKind::kEncoderAttention,' in source.read_text()
        else ['LayoutConvertTaskBody', 'ServingGemmTaskBody::RunDm', 'LayerNormTaskBody'])
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
