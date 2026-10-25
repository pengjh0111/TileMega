"""Certify native dense coordinates against generated, typed call sites."""
import json
import re


def validate(manifest, defines):
    c = manifest['native_dense']
    required = dict(phase='decode', pg='native', mode='L1', loop=1, capacity=1088)
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('native dense execution domain differs from its manifest')
    if (defines.get('TILEMEGA_STATIC_DENSE_NATIVE') != '1' or
            defines.get('TILEMEGA_ARCH_ID') != '900' or
            manifest.get('runtime_batch') not in (1, 2, 4, 8, 16) or
            manifest.get('native_origin') not in ('fixed', 'solver', 'historical_ablation') or c.get('schema') != 2):
        raise ValueError('native dense architecture or coordinate schema is invalid')
    historical = manifest.get('historical_ablation')
    if bool(historical) != (manifest.get('native_origin') == 'historical_ablation'):
        raise ValueError('historical controls cannot claim fixed or solver origin')
    if historical and (historical.get('production_eligible') is not False or
                       defines.get('TILEMEGA_SM90R2_HISTORICAL_ABLATION') != '1'):
        raise ValueError('historical controls require an explicit nonproduction identity')
    if (manifest.get('residency') != c['residency'] or 'TILEMEGA_DN_VECTOR_SUMS' in defines or
            int(defines.get('TILEMEGA_SWIGLU_U', '-1')) != c['swiglu_u']):
        raise ValueError('native dense structural coordinates differ from generated macros')


def dispatch(manifest, defines, source):
    validate(manifest, defines)
    match = re.search(r'kNativeCoordinateIdentity\[\]\s*=\s*("(?:\\.|[^"\\])*");', source)
    if match is None or json.loads(json.loads(match[1])) != manifest['native_dense']:
        raise ValueError('native dense coordinate identity differs from generated source')
    if re.search(r'\b(?:GemmInvocation|StageDesc|ServingGemmTaskBody|ServingGemvTaskBody|'
                 r'tilemega_l1_kernel|tilemega_l1_loop_kernel|tilemega_l2_kernel)\b', source):
        raise ValueError('native dense generated source contains a legacy execution path')
    placements = {}
    for number, body, offset, prefix, l2, late in re.findall(
            r'using Stage_(\d+)\s*=\s*Placement<Body_(\d+),(\d+),(\d+),(\d+)(?:,(true|false))?>;', source):
        if number != body or number in placements:
            raise ValueError('native stage placement is ambiguous')
        placements[number] = (*map(int, (offset, prefix, l2)), late == 'true')

    def placement(number, coordinates, typed):
        expected = tuple(coordinates.get(k, 0) for k in
                         ('workspace_offset', 'preissue_smem_bytes', 'prefetch_l2_bytes'))
        expected += (coordinates.get('preissue_late', False),)
        if ((typed == 'Body' and placements.get(number) != expected) or
                (typed == 'Stage' and any(expected))):
            raise ValueError('native stage placement differs from coordinate identity')

    aliases = re.findall(r'using (Stage|Body)_(\d+)\s*=\s*Gemm<([^;]+)>;', source)
    gemms = manifest['gemms']
    if len(aliases) != len(gemms):
        raise ValueError('native dense GEMM call count differs from manifest')
    result = []
    for index, (gemm, alias) in enumerate(zip(gemms, aliases)):
        typed, number, raw = alias
        args = raw.split(',')
        # ..., op, U, InputNorm, W, C, D, index, register stages, tensor-map index.
        if len(args) != 17:
            raise ValueError('native dense GEMM call signature is invalid')
        register_stages = int(args[15])
        if args[10] != ('true' if gemm['family'] in ('qkv', 'gate_up', 'lm_head') else 'false'):
            raise ValueError('native dense GEMM InputNorm differs from its family')
        batch, n, k, tn, tk, slots, wn, wk = map(int, args[:8])
        actual = dict(tile_m=16, tile_n=tn, tile_k=tk, stages=slots,
                      warps_n=wn, warps_k=wk, split_k=1)
        c = manifest['native_dense']['gemms'][gemm['family']]
        placement(number, c, typed)
        if (gemm['index'] != index or gemm.get('impl') != 'native_mma_sync' or
                batch != manifest['runtime_batch'] or gemm['n'] != n or gemm['k'] != k or
                any(gemm[key] != value for key, value in actual.items()) or
                (tn, tk*tn*2, slots, wn, wk) !=
                (c['tile_n'], c['slot_bytes'], c['slots'], c['warps_n'], c['warps_k']) or
                c['mma'] != 'mma_sync' or int(args[9]) != manifest['native_dense']['swiglu_u'] or
                register_stages != c.get('register_stages', 1) or
                register_stages != gemm.get('register_stages', 1)):
            raise ValueError(f'native dense GEMM call differs from manifest at GEMM {index}')
        op = dict(qkv='kStore', o='kResidual', gate_up='kSwiGLU',
                  down='kResidual', lm_head='kArgmaxPartial')[gemm['family']]
        if args[8] != 'backend::ServingEpilogueOp::' + op:
            raise ValueError('native dense GEMM epilogue differs from its family')
        row = dict(index=index, compiled_variant=index, impl='native_mma_sync', **actual)
        row['register_stages'] = register_stages
        result.append(row)
    if manifest.get('native_origin') == 'historical_ablation':
        attention = re.findall(r'using Body_(\d+)\s*=\s*HistoricalAttention<([^;]+)>;', source)
        merge = re.findall(r'using MergeBody_(\d+)\s*=\s*HistoricalMerge<([^;]+)>;', source)
        h = manifest['historical_ablation']
        if (not attention or len(attention) != len(merge) or
                int(defines.get('TILEMEGA_ATTENTION_IMPL', '-1')) != h['attention_impl'] or
                re.search(r'using (?:Stage|Body)_\d+\s*=\s*Attention<', source)):
            raise ValueError('historical attention identity does not match its call sites')
        for (number, raw), (merge_number, merge_raw) in zip(attention, merge):
            args = list(map(int, raw.split(',')))
            if (len(args) != 16 or args[0] != manifest['runtime_batch'] or
                    args[4:6] != [h['attention_kv_block'], manifest['capacity']] or
                    tuple(args[1:4]) not in ((128,2,1),(64,4,0),(128,3,0),(128,4,0))):
                raise ValueError('historical attention shape differs from its manifest')
            expected = [*args[:3],*args[4:6],args[14],args[15],args[9]]
            if number != merge_number or list(map(int,merge_raw.split(','))) != expected:
                raise ValueError('historical merge differs from its attention stage')
        return result
    attention = re.findall(r'using (Stage|Body)_(\d+)\s*=\s*Attention<([^;]+)>;', source)
    if not attention:
        raise ValueError('native dense attention call is missing')
    c = manifest['native_dense']['attention']
    expected = [c[k] for k in ('chunks', 'block_keys', 'slots', 'new_row_work')]
    shape = None
    for typed, number, raw in attention:
        placement(number, c, typed)
        args = raw.split(',')
        # batch, Hkv, D, Q, norm, chunks, keys, slots, row work, capacity, 11 buffers, K/V maps.
        if len(args) != 23:
            raise ValueError('native dense attention call signature is invalid')
        batch, hkv, dim, queries, norm, chunks, keys, slots, row, capacity = map(int, args[:10])
        current = (dim, queries, norm)
        if (batch != manifest['runtime_batch'] or capacity != manifest['capacity'] or
                [chunks, keys, slots, row] != expected or
                current not in ((128, 2, 1), (64, 4, 0), (128, 3, 0), (128, 4, 0)) or
                (shape is not None and shape != current)):
            raise ValueError('native dense attention call differs from manifest')
        shape = current
    return result


def implementations(manifest):
    c = manifest['native_dense']
    if manifest.get('native_origin') == 'historical_ablation':
        h = manifest['historical_ablation']
        return dict(execution_arch_id=900, program='historical_b_ablation', production_eligible=False,
            dn_vector_sums=c['dn_vector_sums'],swiglu_interleave_u=c['swiglu_u'],
            gemms=[dict(index=g['index'],impl=g['impl'],task_body='Sm90DenseGemm') for g in manifest['gemms']],
            attention=dict(impl='simt' if h['attention_impl']==3 else 'mma16',
                task_body='IndependentAttentionTaskBody',partition='capacity',merge='stage',
                block_extent=h['attention_kv_block']))
    return dict(execution_arch_id=900, program='sm90_dense_native',
                input_norm='consumer_rms', swiglu_interleave_u=c['swiglu_u'],
                gemms=[dict(index=g['index'], impl=g['impl'],
                            task_body='Sm90DenseGemm') for g in manifest['gemms']],
                attention=dict(impl='native_tma_mma_sync', task_body='Sm90DenseAttentionBody',
                               chunks=c['attention']['chunks'],
                               merge='last_arriver' if c['attention']['chunks'] > 1 else 'none',
                               buffers=c['attention']['slots'], partition='actual_past'))
