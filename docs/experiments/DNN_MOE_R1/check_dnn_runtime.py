#!/usr/bin/env python3
"""Build an immutable native DNN runtime fixture, without timing it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--support', action='store_true')
    parser.add_argument('--arch', type=int, default=89)
    parser.add_argument('--embedding', type=int, choices=(0, 1), default=0)
    parser.add_argument('--paged', type=int, choices=(0, 1), default=0)
    parser.add_argument('--split', type=int, choices=(1, 5), default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'framework'))
    from capture_macros_dm import capture
    from identity_dm import resources, sha
    preparation = json.loads((root / 'preparation.json').read_text())
    def unchanged():
        for name, value in preparation['inputs'].items():
            if sha(name) != value:
                raise ValueError('frozen runtime input changed: ' + name)
    unchanged()
    common = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2', '-lineinfo',
              '--expt-relaxed-constexpr', '-UNDEBUG',
              '-I' + str(root / 'include'),
              '-I' + str(root / 'third_party/cutlass/include'),
              '-I' + str(root / 'third_party/cutlass/tools/util/include')]
    sources = json.loads((root / 'support_sources.json').read_text())
    if args.support:
        builds = []
        for index, name in enumerate(sources):
            obj = root / f'support-{index}.o'
            command = [*common, '-arch=sm_80', '-DTILEMEGA_ARCH_ID=800',
                       '-DTILEMEGA_DM_SUPPORT=1', '-x', 'cu', '-c',
                       str(root / name), '-o', str(obj)]
            log = obj.with_suffix('.log')
            with log.open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            capture(command, obj.with_suffix('.macros'))
            builds.append(dict(command=command, object=str(obj), sha256=sha(obj),
                               log_sha256=sha(log)))
        unchanged()
        (root / 'support.json').write_text(json.dumps(builds, indent=2) + '\n')
        return
    support = json.loads((root / 'support.json').read_text())
    for row in support:
        assert sha(row['object']) == row['sha256']
    tag = f'emb{args.embedding}-pg{args.paged}-split{args.split}-sm_{args.arch}'
    binary, source = root / tag, root / 'dm_dnn_runtime_test.cu'
    command = [*common, f'-arch=sm_{args.arch}', f'-DTILEMEGA_ARCH_ID={args.arch * 10}',
               f'-DDM_TEST_EMBEDDING={args.embedding}', f'-DDM_TEST_PAGED={args.paged}',
               f'-DDM_TEST_SPLIT={args.split}', '-Xptxas=-v,-warn-spills',
               str(source), *(row['object'] for row in support), '-o', str(binary)]
    log = root / (tag + '.build.log')
    with log.open('w') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    capture(command, root / (tag + '.macros'))
    unchanged()
    identity = dict(schema='tilemega.dm1.native-test.identity.v1', evidence='verified',
        scope='Native forward ABI and executor integration of DNN primitives; no exported-model or performance claim',
        source=preparation, cu_sha256=sha(source), binary_sha256=sha(binary),
        compiler=dict(path=command[0], sha256=sha(command[0]),
                      version=subprocess.check_output([command[0], '--version'], text=True)),
        command=command, target_arch=f'sm_{args.arch}',
        execution=dict(phase='forward', modes=['L1', 'L2'], pg=bool(args.paged),
                       embedding=bool(args.embedding), split=args.split),
        implementations=['ModelHarness', 'ServingRuntime', 'DmStageRunner',
            'LayerNormTaskBody', 'EmbeddingSumTaskBody' if args.embedding else
            'LayoutConvertTaskBody + ServingConv'],
        support=support, resources=resources(log.read_text()),
        complete_macros=dict(path=str(root / (tag + '.macros/capture.json')),
                             sha256=sha(root / (tag + '.macros/capture.json'))))
    identity['artifact_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    Path(str(binary) + '.identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(dict(event='native_runtime_build_complete', tag=tag,
                         artifact_id=identity['artifact_id'])), flush=True)


if __name__ == '__main__':
    main()
