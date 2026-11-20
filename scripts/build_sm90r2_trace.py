#!/usr/bin/env python3
"""Add stage-boundary timestamps to an immutable SM90R2 library."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--native-phases', action='store_true')
    p.add_argument('--stage-level', type=int, choices=(1, 2), default=1,
                   help='2 adds the D0 in-stage marks (R5 A0)')
    p.add_argument('--sm80-moe-service', action='store_true',
                   help='bind an explicitly instrumented SM80 MoE source tree')
    p.add_argument('--source-root', type=Path,
                   help='checkout of the recorded head when the original tree has moved on')
    a = p.parse_args()
    original, output = a.input.resolve(), a.out.resolve()
    runs = (ROOT / 'runs').resolve()
    allowed = ('sm90r2', 'sm90r3', 'sm90r4', 'sm90r5', 'sm90r6', 'sm90x1', 'a800')
    if not any(output.is_relative_to(runs / r) for r in allowed) or output == original:
        p.error('diagnostic outputs belong under runs/' + ', runs/'.join(allowed))
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.build.identity import generate, verify, validate_snapshot_inputs, digest, snapshot
    identity = verify(original)
    source = dict(identity['source'])
    recorded = source['root']
    if a.sm80_moe_service and (not a.source_root or a.stage_level != 2
                             or identity['arch'] != 'sm_80' or not identity['plan'].get('native_dense', {}).get('expert_fuse')):
        p.error('SM80 MoE service requires a level-2 trace and a fused SM80 reference')
    if a.source_root and not a.sm80_moe_service:
        # Same bytes, other path: every recorded source hash must match there.
        source.update(root=str(a.source_root.resolve()))
        source.pop('compiler', None)
    if not a.sm80_moe_service:
        validate_snapshot_inputs(source['root'], source)
    if identity['trace'] or output.exists():
        raise ValueError('a new trace output and an uninstrumented input are required')
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.cu', '.plan.json', '.source.json'):
        shutil.copy2(str(original) + suffix, str(output) + suffix)
    if a.sm80_moe_service:
        source = snapshot(a.source_root, Path(str(output) + '.source.json'))
        expected = identity['source']['source_files']
        current = source['source_files']
        changed = {p for p in expected.keys() | current.keys() if expected.get(p) != current.get(p)}
        if changed != {
            'include/tilemega/Backend/Sm90DenseMark.cuh',
            'include/tilemega/Backend/Sm90DenseMoe.cuh',
            'include/tilemega/Codegen/tasks/Sm90DenseRuntime.cuh',
        }:
            raise ValueError(f'expected only SM80 fused expert instrumentation, got {sorted(changed)}')
    elif a.source_root:
        source_manifest = json.loads(Path(str(output) + '.source.json').read_text())
        source_manifest.update(root=source['root'])
        source_manifest.pop('compiler', None)
        Path(str(output) + '.source.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    # Worktree paths reach runs/ through a symlink; compare resolved paths.
    command = [str(Path(x).resolve()) if x.endswith(('.so', '.cu')) else x
               for x in shlex.split(Path(str(original) + '.build_command.txt').read_text())]
    sources = [x for x in command if x.endswith('.cu')]
    # Some R1 libraries were compiled from a since-removed staging copy of the
    # recorded source; the copy beside the library must then match its identity.
    if command.count(str(original)) != 1 or len(sources) != 1 or (
            sources[0] != str(original) + '.cu' and
            sha(str(original) + '.cu') != identity['generated_source_sha256']):
        raise ValueError('ambiguous build command')
    command = [str(output) if x == str(original) else str(output) + '.cu'
               if x == sources[0] else x for x in command]
    if a.source_root:
        command = [x if x in (str(output),str(output)+'.cu') else
                   x.replace(recorded,source['root']) for x in command]
    command[1:1] = ['-lineinfo', f'-DTILEMEGA_TRACE_STAGE={a.stage_level}',
                    '-DTILEMEGA_TRACE_STEP=1']
    if a.native_phases:
        command.insert(1, '-DTILEMEGA_TRACE_NATIVE_PHASES=1')
    Path(str(output) + '.build_command.txt').write_text(shlex.join(command) + '\n')
    with Path(str(output) + '.ptxas.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_sm90r1_build.py'),
            '--receipt',str(output)+'.build_limits.json','--max-gib','48','--timeout','5400',
            '--lock','/tmp/tilemega_sm90r2_native_build.lock','--',*command],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    result = generate(output, source['root'], str(output) + '.source.json')
    if not result['trace']:
        raise ValueError('trace identity was not recorded')
    result['performance_eligible'] = False
    result['artifact_id'] = digest({k: v for k, v in result.items() if k != 'artifact_id'})
    Path(str(output) + '.identity.json').write_text(json.dumps(result, indent=2) + '\n')
    record = dict(reference_artifact_id=identity['artifact_id'], trace_artifact_id=result['artifact_id'],
                  performance_eligible=False, native_phases=a.native_phases,
                  sm80_moe_service=a.sm80_moe_service,
                  method='same generated program; boundary timestamps and optional native phase probes')
    Path(str(output) + '.diagnostic.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
