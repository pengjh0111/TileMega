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
            manifest.get('native_origin') not in ('fixed', 'solver') or c.get('schema') != 3):
        raise ValueError('native dense architecture or coordinate schema is invalid')
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
    for number, body, offset, prefix in re.findall(
            r'using Stage_(\d+)\s*=\s*Placement<Body_(\d+),(\d+),(\d+)>;', source):
        if number != body or number in placements:
            raise ValueError('native stage placement is ambiguous')
        placements[number] = (int(offset), int(prefix))
    program = re.search(r'using NativeProgram\s*=\s*Program<(?:\d+,){5}(\d+),', source)
    arena = int(program[1]) if program else None
    streaming = manifest['native_dense']['weight_stream'] is True

    def placement(number, typed, stream, full_width=False):
        # D5 arena i % 2 (attention may take the whole budget); D6 streams
        # either the rule's first slots or nothing (prompt §6).
        found = placements.get(number)
        offsets = {0} if full_width else {(int(number) % 2) * arena}
        if (typed != 'Body' or found is None or arena is None or found[0] not in offsets or
                found[1] not in ((0, stream) if streaming else (0,))):
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
        placement(number, typed, min(slots, k // tk) * tk * tn * 2)
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
    attention = re.findall(r'using (Stage|Body)_(\d+)\s*=\s*Attention<([^;]+)>;', source)
    if not attention:
        raise ValueError('native dense attention call is missing')
    a = manifest['native_dense']['attention']
    expected = [a[k] for k in ('chunks', 'block_keys', 'slots', 'new_row_work')]
    shape = None
    for typed, number, raw in attention:
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
        # Sm90DenseLayout::Attention; storage beyond one arena takes the budget.
        merge = 16*queries*dim + 32*queries + 4*max(chunks, 4)*queries + 4
        storage = (4*slots*keys*dim + 96*dim + merge + 1023) // 1024 * 1024
        full = arena is not None and storage > arena
        placement(number, typed, 0 if full else slots*4*keys*dim, full)
    return result


def implementations(manifest):
    c = manifest['native_dense']
    return dict(execution_arch_id=900, program='sm90_dense_native',
                input_norm='consumer_rms', swiglu_interleave_u=c['swiglu_u'],
                gemms=[dict(index=g['index'], impl=g['impl'],
                            task_body='Sm90DenseGemm') for g in manifest['gemms']],
                attention=dict(impl='native_tma_mma_sync', task_body='Sm90DenseAttentionBody',
                               chunks=c['attention']['chunks'],
                               merge='last_arriver' if c['attention']['chunks'] > 1 else 'none',
                               buffers=c['attention']['slots'], partition='actual_past'))
