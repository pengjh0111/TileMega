# SPDX-License-Identifier: BSD-3-Clause
"""Closed ResNet proof separating selected im2col loops from retained fallbacks."""
import bisect
import hashlib
from pathlib import Path
import re

# This closed CFG proof must be reviewed again when these runtime sources change.
APPROVED_SOURCE_HASHES = {
    "include/tilemega/Backend/DmMoeOperands.h": "bc90c3670ee912cc791ff254b4108a8c596e884d1d163a97a96f8724f4d7cbff",
    "include/tilemega/Backend/Sm90Conv.h": "af1a2fb842cd4a936717ca650801abb8fcda1fb4fc9c526dcac5ada3407248a9",
    "include/tilemega/Codegen/DmDescriptors.h": "2f9ec46647624f9721141d9e9e9058ac7a5f3b1b1b4e0d88bb74584983c04b05",
    "include/tilemega/Codegen/tasks/GemmStageTaskBody.h": "532a6bcd1d997eb4fd219574117b7e1d2f96de77459d9cf6106558ac2d477436",
    "include/tilemega/Codegen/tasks/ModelHarness.cuh": "eed0d952c4f1f7fdf33d922a12a1973c678009d379700deeca05bd8e72cfbadc",
    "include/tilemega/Codegen/tasks/ModelRuntime.h": "9d7e21dc7e33ed19cbb6e1db3ad651eb7ab5c90138b1d2b65a5b20fa9f441554",
    "include/tilemega/Codegen/tasks/ServingGemmTaskBody.h": "5c29c60633076fdf13170b455ae2c5421d490ec807de3a1901397fd0a067278e"
}


def source_gate(root):
    result = {}
    for name, expected in APPROVED_SOURCE_HASHES.items():
        actual = hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('selected-path CFG source changed; new proof required: ' + name)
        result[name] = actual
    return result


def table(source, kind, name):
    match = re.search(r'constexpr\s+'+kind+r'\s+'+name+r'\[\]\s*=\s*\{(.*?)\n\};', source, re.S)
    if match is None:
        raise ValueError('selected-path descriptor table missing: ' + name)
    return [line.strip() for line in match[1].splitlines() if line.strip()]


def number(text):
    return int(re.sub(r'(?:ull|llu|ul|u)$', '', text.strip()))


def selected_conv_routes(identity, generated):
    plan, defines = identity['plan'], identity['defines']
    if (plan.get('model') != 'resnet18' or plan.get('phase') != 'forward' or
        plan.get('batch_lo') != 1 or plan.get('batch_hi') != 1 or plan.get('seq') != 1 or
        plan.get('pg') != 'l2' or int(defines.get('TILEMEGA_ARCH_ID', 0)) != 900 or
        int(defines.get('TILEMEGA_CONV_TMA_IM2COL', 0)) != 1 or
        int(defines.get('TILEMEGA_DM_WGMMA_INLINE', 0)) != 1):
        raise ValueError('selected-path proof only covers the fixed native ResNet B1 grid')
    gemms = table(generated, 'GemmDesc', 'kGemms')
    convs = [[number(x) for x in re.findall(r'\d+(?:ull|u)?', line)]
             for line in table(generated, 'ConvDesc', 'kConvolutions')]
    runtime = table(generated, 'GemmRuntimeDesc', 'kRuntimeGemms0')
    if len(gemms) != 21 or len(runtime) != 21 or len(plan['gemms']) != 21:
        raise ValueError('complete ResNet GEMM matrix is required')
    selected = []
    for index, (entry, descriptor, actual) in enumerate(zip(plan['gemms'], gemms, runtime)):
        variant, split, tm, tn, tk, stages = map(number, re.findall(r'\d+(?:u)?', actual))
        if entry['index'] != index or any(entry[key] != value for key, value in
                zip(('split_k', 'tile_m', 'tile_n', 'tile_k', 'stages'), (split, tm, tn, tk, stages))):
            raise ValueError('manifest and actual runtime geometry differ')
        implementation = int(defines.get(f'TILEMEGA_GEMM_IMPL_{variant}', 0))
        if implementation not in (0, 1, 2) or entry['impl'] != ('mma16', 'gemv', 'wgmma')[implementation]:
            raise ValueError('manifest and actual runtime implementation differ')
        if entry['impl'] != 'wgmma':
            continue
        shape = re.match(r'\{(\d+),\s*(\d+),', descriptor)
        access = re.search(r'\{static_cast<DmAAccess>\((\d+)u\),\s*static_cast<DmBAccess>\((\d+)u\),\s*([^{}]+),\s*\{static_cast<DmWriteKind>', descriptor)
        if shape is None or access is None:
            raise ValueError('actual GEMM/access descriptor is not recognized')
        n, k = map(int, shape.groups())
        fields = [int(access[1]), int(access[2])] + [number(x) for x in access[3].split(',')]
        if len(fields) != 10 or fields[0] != 1 or fields[1] != 0 or fields[8] != 4294967295 or fields[9] != 0:
            raise ValueError('real dense/gather/scaled WG is outside the im2col proof')
        if fields[5] >= len(convs):
            raise ValueError('convolution descriptor is outside its table')
        conv = convs[fields[5]]
        binding = r'template<>\s+struct\s+DmGemmBinding<'+str(index)+r',\s*'+str(variant)+r'>\s*\{\s*static constexpr unsigned roles = 1u;\s*\};'
        if (len(conv) != 17 or split != 1 or tm != 64 or tn not in (64, 128) or
            tk != 64 or stages != 2 or conv[0] != 1 or conv[3] < 64 or conv[3] % 64 or
            conv[3]*conv[5]*conv[6] != k or conv[4] != n or k <= 0 or k % 64 or
            re.search(binding, generated) is None):
            raise ValueError('split, channel tail, fusion or iteration geometry is outside the proof')
        selected.append(dict(index=index, compiled_variant=variant, access_a='im2col',
            access_b='dense', a_scale=None, chunks=1, k_begin=0, k_count=k,
            iterations=k//64, conv_index=fields[5], conv_c=conv[3],
            dm_enabled=True, backend='Sm90Conv', dense_WG_reachable=False,
            untyped_or_partial_reachable=False))
    if [row['index'] for row in selected] != list(range(1, 20)):
        raise ValueError('selected WG set differs from the nineteen ResNet convolutions')
    return selected


def selected_loop_proof(identity, generated, audit, sass_lines, source_root):
    """Require all potentially selected im2col K backedges to contain zero locals."""
    fingerprints = source_gate(source_root)
    routes = selected_conv_routes(identity, generated)
    functions = {}
    kinds = ('tilemega_l1_kernel', 'tilemega_l1_loop_kernel',
             'tilemega_l2_kernel', 'tilemega_stage_kernel')
    if any(sum(kind in row['function'] for row in audit['functions']) != 1 for kind in kinds):
        raise ValueError('all four actual executor kernels must be audited')
    for function in audit['functions']:
        regions = sorted(function['regions'], key=lambda row: int(row['begin'], 16))
        nodes = [dict(begin=row['begin'], end=row['end'],
            local_instructions=len(row['local_instructions']), HGMMA=0, TMA4D=0) for row in regions]
        for first, second in zip(nodes, nodes[1:]):
            if int(first['end'], 16) >= int(second['begin'], 16):
                raise ValueError('overlapping conservative loops need a new CFG audit')
        functions[function['function']] = ([int(row['begin'], 16) for row in nodes], nodes)
    current = None
    total_hgmma = 0
    for line in sass_lines:
        match = re.search(r'Function : (\S+)', line)
        if match:
            current = functions.get(match[1])
            continue
        match = re.search(r'/\*([0-9a-fA-F]+)\*/\s+([^;]+);', line)
        if match is None or current is None:
            continue
        address, instruction = int(match[1], 16), match[2]
        total_hgmma += int('HGMMA' in instruction)
        starts, nodes = current
        index = bisect.bisect_right(starts, address)-1
        if index >= 0 and address <= int(nodes[index]['end'], 16):
            nodes[index]['HGMMA'] += int('HGMMA' in instruction)
            nodes[index]['TMA4D'] += int('UTMALDG.4D' in instruction)
    active, inactive = [], []
    for name, (_, nodes) in functions.items():
        im2col = [row for row in nodes if row['TMA4D']]
        dense = [row for row in nodes if not row['TMA4D']]
        if len(im2col) != 20 or len(dense) != 21 or any(row['HGMMA'] != 4 for row in nodes):
            raise ValueError('binary loop structure differs from the proved ResNet CFG')
        if any(row['local_instructions'] for row in im2col):
            raise ValueError('selected im2col K loop contains local-memory traffic')
        active += [dict(function=name, **row) for row in im2col]
        inactive += [dict(function=name, **row) for row in dense]
    covered = sum(row['HGMMA'] for row in active+inactive)
    if len(functions) != 4 or covered != total_hgmma or total_hgmma != audit['counts']['HGMMA']:
        raise ValueError('selected-path audit does not cover all compiled HGMMA')
    return dict(passed=True, scope='fixed ResNet18 B1 native im2col candidate only',
        evidence='CFG reachability inferred from verified fixed descriptors and pinned source; GPU gates required',
        source_fingerprints=fingerprints, selected_runtime_routes=routes,
        selected_K_loop_local_instructions=0, im2col_K_loops=active,
        unreachable_dense_WG_K_loops=inactive,
        conservative_whole_generic_local_instructions=audit['hot_local_instructions'],
        whole_binary_zero_spill_claim=False,
        reason='chunks1 and im2col-enabled typed dispatch exclude partial/untyped and dense WG branches')
