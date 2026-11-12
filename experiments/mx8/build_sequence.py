"""Build opt-in stage sequences from an immutable main-experiment CU."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import digest, generate, sha


def replace_body(text, start, body):
    opening = text.index('{', start)
    depth = 1
    closing = opening + 1
    while depth:
        depth += (text[closing] == '{') - (text[closing] == '}')
        closing += 1
    return text[:opening] + '{\n' + body + '\n}' + text[closing:]


def transform_runtime(text):
    start = text.index('inline int Launch(')
    text = replace_body(text, start, '  (void)step_ns;\n  return Mx8Launch(s, first, steps, iteration, stream);')
    target = '  s->steps = count;\n  return 0;'
    if text.count(target) != 1:
        raise ValueError('unknown set_steps layout')
    text = text.replace(target, '  s->steps = count;\n  return Mx8Prepare(s);')
    marker = '  if (!s) return;\n'
    if text.count(marker) != 1:
        raise ValueError('unknown Free layout')
    return text.replace(marker, marker + '  Mx8Release(s);\n')


def build(source, out, mode):
    source = source.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    binary = out / 'plan.so'
    if binary.exists():
        raise ValueError('build output already exists; retain it and use another directory')
    original = source.read_text()
    moe = '#include <tilemega/Codegen/tasks/mx/MoeProgramV3.cuh>' in original
    runtime = 'MoeRuntimeV3.cuh' if moe else 'DecodeRuntimeV2.cuh'
    before = ROOT / 'include/tilemega/Codegen/tasks/mx' / runtime
    overlay = out / runtime
    overlay.write_text(transform_runtime(before.read_text()))
    declaration = '''#include <cstdint>
template <class S> int Mx8Prepare(S*);
template <class S> void Mx8Release(S*);
template <class S> int Mx8Launch(S*, unsigned, unsigned, unsigned long long, void*);
'''
    marker = '#include <tilemega/Codegen/tasks/mx/' + runtime + '>'
    if original.count(marker) != 1:
        raise ValueError('unknown generated runtime include')
    generated = original.replace(marker, declaration + '#include "' + str(overlay) + '"\n' +
        '#include "' + str(ROOT / 'experiments/mx8/sequence.cuh') + '"')
    Path(str(binary) + '.cu').write_text(generated)
    manifest = json.loads(Path(str(source)[:-3] + '.plan.json').read_text())
    manifest.update(ablation='AB4c', sequence_mode=mode, baseline_reference='frozen_main_experiment',
        preserves_selected_geometry=True, materializes_route_table=moe, watchdog=0,
        ablation_original_source=dict(path=str(source), sha256=sha(source)))
    Path(str(binary) + '.plan.json').write_text(json.dumps(manifest, indent=2) + '\n')
    inputs = [p for p in (ROOT / 'include').rglob('*') if p.is_file() and p.suffix in ('.h', '.cuh', '.hpp')]
    inputs += [ROOT / 'experiments/mx8/sequence.cuh', Path(__file__).resolve(), overlay, source]
    files = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    compiler = '/opt/maca/tools/cu-bridge/bin/cucc'
    snapshot = dict(root=str(ROOT), head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_files=files, source_digest=digest(files), compiler=compiler, compiler_sha256=sha(compiler))
    source_json = Path(str(binary) + '.source.json')
    source_json.write_text(json.dumps(snapshot, indent=2) + '\n')
    command = [compiler, '-std=c++20', '-O3', '-Xcompiler', '-resource-usage', '-fno-strict-aliasing',
        '-mllvm', '--max-mtreg-number=255', '-DTILEMEGA_MX_STATIC_DECODE_V2=1', '-DTILEMEGA_WATCHDOG=0',
        '-DTILEMEGA_TRACE_STAGE=0', '-DTILEMEGA_MX_DIAG=0', '-DMX8_MOE=' + str(int(moe)),
        '-DMX8_HOST_LAUNCH=' + str(int(mode == 'hostlaunch'))]
    if moe:
        command += ['-DTILEMEGA_SYNC_V3=1', '-DTILEMEGA_MX_STATIC_MOE_V3=1']
    command += ['-shared', '-Xcompiler=-fPIC', '-x', 'cu', '-I' + str(ROOT / 'include'),
        str(binary) + '.cu', '-L/opt/maca/lib', '-lcudart', '-o', str(binary)]
    Path(str(binary) + '.build_command.txt').write_text(shlex.join(command) + '\n')
    with Path(str(binary) + '.ptxas.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    identity = generate(binary, ROOT, source_json)
    print(json.dumps(dict(binary=str(binary), artifact_id=identity['artifact_id'], mode=mode)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['graph', 'hostlaunch'], required=True)
    args = parser.parse_args()
    build(args.source, args.out, args.mode)
