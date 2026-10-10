#!/usr/bin/env python3
"""Freeze CG-generated CUDA and its runtime before correctness queue submission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-export', type=Path)
    parser.add_argument('--elementwise',action='store_true',
        help='use the operator-fixture BF16 tolerance instead of model output metrics')
    parser.add_argument('--bridge', type=Path)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--architectures', type=int, nargs='+', default=[89],
        choices=[80,89,90,100,120])
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--sanitizers', nargs='*', default=[], choices=['memcheck','racecheck'])
    parser.add_argument('--diagnostic',action='store_true',
        help='separate sm89 intermediate-copy diagnostic; not a correctness gate')
    parser.add_argument('--input-tensors',type=Path)
    parser.add_argument('--moe-bridge',type=Path)
    parser.add_argument('--moe-checkpoint',type=Path)
    parser.add_argument('--moe-hidden',type=Path)
    parser.add_argument('--moe-decoder-config',type=Path,
        help='seeded complete decoder fixture; ineligible for the real-weight G-MOE gate')
    parser.add_argument('--moe-decoder-bridge',type=Path,
        help='enable native-input component checks for the complete decoder fixture')
    args = parser.parse_args()
    if args.processes<1:
        parser.error('--processes must be positive')
    if bool(args.model_export)!=bool(args.bridge) or not 1<=args.batch<=64:
        parser.error('model checks require both export and bridge, with batch in [1,64]')
    if args.elementwise and not args.model_export:
        parser.error('--elementwise requires an exported operator fixture')
    if args.diagnostic and not args.model_export:
        parser.error('intermediate diagnostics require an upstream model export')
    if args.input_tensors and not args.diagnostic:
        parser.error('supplied inputs require the separate diagnostic artifact')
    if args.moe_bridge and (args.model_export or args.diagnostic or args.input_tensors):
        parser.error('MoE region validation uses its own bridge and inputs')
    if bool(args.moe_checkpoint)!=bool(args.moe_hidden) or (args.moe_checkpoint and not args.moe_bridge):
        parser.error('real MoE checks require --moe-bridge, --moe-checkpoint and --moe-hidden')
    if args.moe_decoder_config and (args.moe_bridge or args.model_export or args.diagnostic):
        parser.error('complete decoder fixtures use their own config and checker')
    if args.moe_decoder_bridge and not args.moe_decoder_config:
        parser.error('--moe-decoder-bridge requires --moe-decoder-config')
    repo = Path(__file__).resolve().parents[3]
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'framework').mkdir()
    for name in ['capture_macros_dm.py', 'identity_dm.py']:
        shutil.copy2(Path(__file__).with_name(name), root/'framework'/name)
    shutil.copy2(Path(__file__).with_name('check_generated_dnn.py'), root/'check.py')
    shutil.copy2(args.source, root/'generated.cu')
    if args.moe_decoder_config:
        shutil.copy2(args.moe_decoder_config,root/'decoder_config.json')
        text=(root/'generated.cu').read_text()
        if re.search(r'^#define TILEMEGA_SERVING_SEQ 1$',text,re.M):
            (root/'generated.cu').write_text('#define TILEMEGA_SERVING_PAST_LO 3\n#define TILEMEGA_SERVING_PAST_HI 3\n'+text)
    if args.input_tensors:
        shutil.copy2(args.input_tensors,root/'inputs.safetensors')
    if args.diagnostic or args.moe_bridge or args.moe_decoder_config:
        with (root/'generated.cu').open('a') as stream:
            stream.write('''
// Test-only accessor, appended to a separate diagnostic artifact.
extern "C" void* tm_dm_debug_buffer(void* handle, unsigned index,
                                    unsigned long long* metadata) {
  auto* plan=static_cast<tilemega::codegen::serving::Plan*>(handle);
  if(!plan || index>=plan->model.spec->buffer_count || !metadata) return nullptr;
  auto const& layout=plan->model.spec->buffers[index].layout;
  metadata[0]=static_cast<unsigned>(layout.kind); metadata[1]=layout.rank;
  for(unsigned i=0;i<4;++i) {
    metadata[2+i]=layout.logical[i]; metadata[6+i]=layout.physical[i];
    metadata[10+i]=layout.strides[i];
  }
  metadata[14]=layout.halo_top; metadata[15]=layout.halo_bottom;
  metadata[16]=layout.halo_left; metadata[17]=layout.halo_right;
  return plan->model.buffers[index];
}
''')
    for name in ['dnn_semantic_lifting_test.cpp', 'dnn_epilogue_semantics_test.cpp']:
        (root/'test/unit').mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/'test/unit'/name, root/'test/unit'/name)
    for name in ['lib/Analysis/TaskStorage.cpp', 'lib/Frontend/DnnStorage.cpp',
                 'lib/Frontend/DnnSemanticLifting.cpp']:
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, root/name)
    for name in ['tilemega/__init__.py', 'tilemega/serving/__init__.py', 'tilemega/serving/plan.py']:
        (root/'python'/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/'python'/name, root/'python'/name)
    shutil.copytree(repo/'python/tilemega',root/'python/tilemega',dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if args.model_export or args.moe_bridge or args.moe_decoder_config:
        if args.moe_bridge or args.bridge or args.moe_decoder_bridge:
            shutil.copy2(args.moe_bridge or args.bridge or args.moe_decoder_bridge,root/'bridge.json')
    shutil.copytree(repo/'include', root/'include')
    for name in ['include', 'tools/util/include']:
        shutil.copytree(repo/'third_party/cutlass'/name, root/'third_party/cutlass'/name)
    support = ['lib/Target/TargetSpec.cpp', 'lib/Support/Json.cpp',
        'lib/Codegen/RuntimeTaskGraph.cpp', 'lib/Solver/PlanMaterialize.cpp',
        'lib/Dialect/CouplingGraph/PlacementPlan.cpp',
        'lib/Solver/BalancedPlacement.cpp', 'lib/Solver/ListScheduler.cpp']
    for name in support:
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, root/name)
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()}
    if args.moe_hidden:
        inputs[str(args.moe_hidden.resolve())]=hashlib.sha256(args.moe_hidden.read_bytes()).hexdigest()
    preparation = dict(source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
        text=True).strip(), diff_sha256=hashlib.sha256(subprocess.check_output(
            ['git', 'diff', 'HEAD'], cwd=repo)).hexdigest(), support=support, inputs=inputs,
        codegen_test_sha256=hashlib.sha256((repo/'test/unit/dnn_semantic_lifting_test.cpp').read_bytes()).hexdigest())
    (root/'preparation.json').write_text(json.dumps(preparation, indent=2)+'\n')
    steps = []
    def step(name, command, after, priority, seconds):
        steps.append(dict(name=name, command=['flock', '/root/r14_work/gpu.lock',
            'timeout', str(seconds), *command], cwd=str(repo), gpu=False,
            after=after, priority=priority, timeout_s=seconds+600))
    for arch in ([89] if args.diagnostic else args.architectures):
        step(f'build_sm_{arch}', ['python3', str(root/'check.py'), '--root', str(root),
            '--arch', str(arch)], [], 0 if arch==89 else 200, 1800)
    command = ['/root/dm1_work/venv-gpu/bin/python', str(root/'check.py'), '--root', str(root)]
    if args.model_export:
        command=['env','PYTHONPATH='+str(root/'python'),'/root/dm1_work/venv-gpu/bin/python',
            '-m','tilemega.dnn.check_generated','--library',str(root/'generated-sm_89.so'),
            '--export',str(args.model_export.resolve()),'--bridge',str(root/'bridge.json'),
            '--batch',str(args.batch),'--out',str(root/'correctness.json')]
    if args.elementwise:
        command+=['--elementwise']
    if args.moe_bridge:
        command=['env','PYTHONPATH='+str(root/'python'),'/root/dm1_work/venv-gpu/bin/python',
            '-m','tilemega.moe.check_generated','--library',str(root/'generated-sm_89.so'),
            '--bridge',str(root/'bridge.json'),'--out',str(root/'correctness.json')]
        if args.moe_checkpoint:
            command+=['--checkpoint',str(args.moe_checkpoint.resolve()),'--hidden',str(args.moe_hidden.resolve())]
    if args.moe_decoder_config:
        command=['env','PYTHONPATH='+str(root/'python'),'/root/dm1_work/venv-gpu/bin/python',
            '-m','tilemega.moe.check_decoder','--library',str(root/'generated-sm_89.so'),
            '--config',str(root/'decoder_config.json'),'--out',str(root/'correctness.json')]
        if args.moe_decoder_bridge:
            command+=['--bridge',str(root/'bridge.json')]
    if args.diagnostic:
        command += ['--diagnostics',str(root/'intermediates.json')]
    if args.input_tensors:
        command += ['--input-tensors',str(root/'inputs.safetensors')]
    for process in range(1 if args.diagnostic else args.processes):
        step(f'check_{process:02}', command, ['build_sm_89' if process==0 else 'check_00'], 10, 300)
    for tool in ([] if args.diagnostic else args.sanitizers):
        step(tool, ['/usr/local/cuda/bin/compute-sanitizer', '--tool', tool,
            '--target-processes', 'all', '--error-exitcode', '86', *command], ['build_sm_89'], 2, 1800)
    (root/'queue').mkdir()
    (root/'queue/queue_generated.json').write_text(json.dumps(steps, indent=2)+'\n')
    print(json.dumps(dict(root=str(root), steps=len(steps))), flush=True)
