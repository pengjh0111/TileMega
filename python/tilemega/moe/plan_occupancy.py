"""Query a native MoE-L24 plan's attributes and occupancy without launching it."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from tilemega import mx5_plan_occupancy as common
from tilemega.mx5_resources import inspect

KERNEL = '_ZN8tilemega7codegen2mx16mx_moe_kernel_v2IN10tm_mx_plan4PlanEEEvNT_7BindingENS1_13StageLaunchV2E'
STUB = KERNEL.replace('16mx_moe_kernel_v2', '31__device_stub__mx_moe_kernel_v2')


def exported_stub(text, version=2):
    if version not in (2,3):
        raise ValueError('unsupported MoE program version')
    stub=STUB if version==2 else STUB.replace('mx_moe_kernel_v2','mx_moe_kernel_v3')
    needle='mx_moe_kernel_v'+str(version)
    rows = [line.split() for line in text.splitlines() if line.split()]
    names = [row[-1] for row in rows]
    if [name for name in names if 'mx_moe_kernel_v' in name] != [stub] or names.count('tm_plan_shared_bytes') != 1:
        raise ValueError('expected exactly the native MoE registered stub and shared query')
    if any('mx_decode_kernel_v2' in name for name in names):
        raise ValueError('MoE query cannot contain a dense kernel')
    for symbol in (stub, 'tm_plan_shared_bytes'):
        row = next(row for row in rows if row[-1] == symbol)
        if len(row) != 3 or row[-2] not in ('T', 'W') or not re.fullmatch('[0-9a-fA-F]+', row[0]):
            raise ValueError('expected an exported host code symbol: ' + symbol)
    return stub


def validate_inputs(spec, *, inspect_resource=None, command=common.run_text):
    if spec.get('kind') != 'mx5_moe_plan_occupancy_inputs' or spec.get('schema_version') != 1:
        raise ValueError('unsupported MoE query input schema')
    threads, batch = spec.get('threads'), spec.get('batch')
    version=spec.get('program_version',2)
    if version not in (2,3) or (version==3 and threads!=512):
        raise ValueError('unsupported MoE program version/thread pair')
    kernel=KERNEL if version==2 else KERNEL.replace('mx_moe_kernel_v2','mx_moe_kernel_v3')
    if type(threads) is not int or threads not in (512, 576) or type(batch) is not int or batch not in (1, 2, 4, 8, 16):
        raise ValueError('unsupported MoE batch/thread shape')
    paths = {name: common.checked(spec[name]) for name in
        ('binary', 'generated_source', 'resources', 'log', 'context', 'source_snapshot')}
    report = json.loads(paths['resources'].read_text())
    actual = (inspect_resource or inspect)(paths['binary'], paths['log'], threads, kernel_name='mx_moe_kernel_v'+str(version))
    if report != actual or report.get('kernel', {}).get('name') != kernel or set(report.get('kernel_instructions', {})) != {kernel} or len(report.get('device_images', [])) != 1:
        raise ValueError('resource report does not describe the exact single MoE kernel')
    source = paths['generated_source'].read_text()
    geometry=spec.get('model_geometry',{})
    olmoe=geometry.get('experts')==64
    if geometry and geometry not in ({'layers':16,'hidden':2048,'intermediate':1024,'experts':64,'hkv':16,'q':1},
                                    {'layers':24,'hidden':2048,'intermediate':768,'experts':128,'hkv':4,'q':8}):
        raise ValueError('unsupported MoE query model geometry')
    for name, expected in dict(kThreads=threads, kBatch=batch, kGrid=104, kLayers=16 if olmoe else 24,
            kHidden=2048, kIntermediate=1024 if olmoe else 768, kCapacity=1088).items():
        if re.findall(r'\b' + name + r'\s*=\s*(\d+)\b', source) != [str(expected)]:
            raise ValueError('generated MoE Plan constant differs: ' + name)
    if olmoe:
        for name,expected in {'kExperts':64,'kQ':1,'kHeadDim':128}.items():
            if re.findall(r'\b'+name+r'\s*=\s*(\d+)\b',source)!=[str(expected)]:
                raise ValueError('OLMoE query source geometry differs: '+name)
        for name,expected in {'kRowQkNorm':'true','kNormalizeTopK':'false'}.items():
            if re.findall(r'\b'+name+r'\s*=\s*(true|false)\b',source)!=[expected]:
                raise ValueError('OLMoE query numeric scope differs: '+name)
    if (f'MoeProgramV{version}.cuh' not in source or f'MoeRuntimeV{version}.cuh' not in source or
            not re.search(r'\bnamespace\s+tm_mx_plan\s*\{', source) or not re.search(r'\bstruct\s+Plan\s*\{', source) or
            'DecodeProgramV2.cuh' in source):
        raise ValueError('requires the native MoE program/runtime')
    context = json.loads(paths['context'].read_text())
    snapshot = json.loads(paths['source_snapshot'].read_text())
    if spec.get('context_kind') == 'compile_receipt':
        for name in ('binary', 'generated_source', 'log', 'source_snapshot'):
            if context.get(name + '_sha256') != spec[name]['sha256']:
                raise ValueError('MoE compile receipt mismatch: ' + name)
        if context.get('kind') != 'mx5_native_compile_receipt' or context.get('returncode') != 0 or context.get('same_compile_mir') != spec['log']['sha256'] or context.get('changed_inputs'):
            raise ValueError('requires a completed unchanged native MoE compile')
    elif spec.get('context_kind') == 'native_identity':
        manifest = common.checked(spec['manifest'])
        paths['manifest'] = manifest
        for field, name in (('binary_sha256', 'binary'), ('generated_source_sha256', 'generated_source'),
                ('resources_sha256', 'log'), ('manifest_sha256', 'manifest')):
            if context.get(field) != spec[name]['sha256']:
                raise ValueError('MoE native identity mismatch: ' + field)
        plan = json.loads(manifest.read_text())
        if context.get('resources', {}).get(kernel) != report['kernel'] or context.get('source') != snapshot:
            raise ValueError('MoE identity resource/source mismatch')
        if plan.get('program_kind') != 'moe' or plan.get('executor') not in ('mx_static', 'mx_static_prompt') or plan.get('threads') != threads or plan.get('kThreads') != threads or plan.get('batch_lo') != batch or plan.get('batch_hi') != batch:
            raise ValueError('MoE manifest batch/thread/native executor differs')
        if plan.get('program_version',2)!=version:
            raise ValueError('MoE manifest program version differs')
    else:
        raise ValueError('unsupported MoE identity/compile context')
    nm = command(['nm', '-D', '--defined-only', str(paths['binary'])])
    stub = exported_stub(nm,version)
    assembly = command(['objdump', '-d', '--disassemble=tm_plan_shared_bytes', str(paths['binary'])])
    shared = common.shared_constant(assembly)
    return dict(spec=spec, paths=paths, report=report, threads=threads, stub=stub,
        expected_shared_bytes=shared, nm_stdout=nm, shared_query_disassembly=assembly)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--query-only', action='store_true', required=True)
    parser.add_argument('--iterations', type=int, choices=[0], required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('output exists; preserve historical evidence')
    guard = common.guard_evidence(os.environ)
    inputs = common.record(args.inputs)
    validated = validate_inputs(json.loads(common.checked(inputs).read_text()))
    # This shared ABI helper only queries the verified host stub; it never
    # calls tm_plan_create, launch, or launch_steps.
    with args.out.open('x') as output:
        runtime, row = None, None
        try:
            runtime = common.record(common.RUNTIME)
            row = common.query(validated, common.Runtime(), common.Library(validated['paths']['binary']), guard)
            for item in [inputs, runtime, *(value for value in validated['spec'].values()
                    if isinstance(value, dict) and 'path' in value and 'sha256' in value)]:
                common.checked(item)
        except Exception as error:
            if row is None:
                row = dict(kind='mx5_plan_occupancy', schema_version=1,
                    binary_sha256=validated['spec']['binary']['sha256'], threads=validated['threads'],
                    resident_ctas=None, query_only=True, iterations=0, no_launch=True, launches=0,
                    pid=os.getpid(), guard=guard, bound_inputs=validated['spec'],
                    registered_host_stub=validated['stub'], execution_authorized=False, grants_512_exception=False)
            row.update(query_status=1, query_status_source='loader_or_identity_error', exception=repr(error))
        row.update(program_kind='moe', inputs_manifest=inputs, runtime_library=runtime,
            helper_source=common.record(Path(__file__)), common_helper=common.record(Path(common.__file__)))
        output.write(json.dumps(row, indent=2) + '\n')
    return 0 if row['query_status'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
