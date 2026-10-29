"""Certify native dense coordinates against generated, typed call sites."""
import json
import re


# Typed stages of one layer (prompt §6 D7, §8); a group holds an even number.
LAYER = dict(dense=('qkv', 'attention', 'o', 'gate_up', 'down'),
             moe=('qkv', 'attention', 'o', 'router', 'expert_gate_up', 'expert_down'))
# MR7: an `expert_fuse` coordinate runs both expert GEMMs as one typed stage.
FUSED_LAYER = ('qkv', 'attention', 'o', 'router', 'expert_fused')
FUSED = ('expert_gate_up', 'expert_down')
DENSE_OPS = dict(qkv='kStore', o='kResidual', gate_up='kSwiGLU', down='kResidual',
                 lm_head='kArgmaxPartial')
MOE_MODES = dict(router='kRouter', expert_gate_up='kGateUp', expert_down='kDown')
ATTENTION = ((128, 2, 1), (64, 4, 0), (128, 3, 0), (128, 4, 0))
# (Hkv, D, Q, norm mode): Qwen3-30B-A3B GQA with per-head norm; OLMoE MHA with the
# whole-row norm (A3).
MOE_ATTENTION = ((4, 128, 8, 1), (16, 128, 1, 2))


def validate(manifest, defines):
    c = manifest['native_dense']
    required = dict(phase='decode', pg='native', mode='L1', loop=1, capacity=1088)
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('native dense execution domain differs from its manifest')
    if (defines.get('TILEMEGA_STATIC_DENSE_NATIVE') != '1' or
            defines.get('TILEMEGA_ARCH_ID') != '900' or
            manifest.get('runtime_batch') not in (1, 2, 4, 8, 16) or
            manifest.get('native_origin') not in ('fixed', 'solver') or
            c.get('schema') not in (5, 6) or
            c.get('model') not in LAYER):
        raise ValueError('native dense architecture or coordinate schema is invalid')
    if (manifest.get('residency') != c['residency'] or 'TILEMEGA_DN_VECTOR_SUMS' in defines or
            int(defines.get('TILEMEGA_SWIGLU_U', '-1')) != c['swiglu_u']):
        raise ValueError('native dense structural coordinates differ from generated macros')


def dispatch(manifest, defines, source):
    validate(manifest, defines)
    c = manifest['native_dense']
    match = re.search(r'kNativeCoordinateIdentity\[\]\s*=\s*("(?:\\.|[^"\\])*");', source)
    if match is None or json.loads(json.loads(match[1])) != c:
        raise ValueError('native dense coordinate identity differs from generated source')
    if re.search(r'\b(?:GemmInvocation|StageDesc|ServingGemmTaskBody|ServingGemvTaskBody|'
                 r'tilemega_l1_kernel|tilemega_l1_loop_kernel|tilemega_l2_kernel)\b', source):
        raise ValueError('native dense generated source contains a legacy execution path')
    fused = c.get('expert_fuse') is True
    if fused and c['model'] != 'moe':
        raise ValueError('native expert fusion needs a MoE coordinate')
    layer = FUSED_LAYER if fused else LAYER[c['model']]
    per_group = 2 if len(layer) % 2 else 1
    size = per_group*len(layer)
    head = size + 1
    placements, group_tiles = {}, {}
    for number, body, offset, prefix, tiles in re.findall(
            r'using Stage_(\d+)\s*=\s*Placement<Body_(\d+),(\d+),(\d+)(?:,(\d+))?>;', source):
        if number != body or number in placements:
            raise ValueError('native stage placement is ambiguous')
        placements[number] = (int(offset), int(prefix))
        if tiles:
            group_tiles[number] = int(tiles)
    # D8: exactly each layer's qkv (typed stages 1, 1 + per layer, ...) waits
    # by KV-group counters, with one tile count.
    qkv = {str(1 + len(layer) * i) for i in range(per_group)}
    if set(group_tiles) != (qkv if c['group_wait'] is True else set()) or \
            len(set(group_tiles.values())) > 1 or any(v <= 0 for v in group_tiles.values()):
        raise ValueError('native group wait differs from its coordinate identity')
    # Program<batch, capacity, residency, buffers, maps, arena, layers, slots,
    # loop, embed, Stage_0, LayerGroup<P, Stage_1..Stage_S>, Stage_{S+1}>
    # (D7, D9): dense groups two layers of five, MoE one layer of six.
    group = ''.join(f',Stage_{i}' for i in range(1, size + 1))
    # ... Stage_{S+1}, cluster> (D10).
    program = re.search(r'using NativeProgram\s*=\s*Program<(?:\d+,){5}(\d+),(\d+),\d+,(true|false),'
                        rf'(true|false),Stage_0,LayerGroup<{per_group}{group}>,Stage_{head},(\d+)>;', source)
    lm = re.search(rf'using Body_{head}\s*=\s*Head<HeadGemm_{head},\d+,\d+,\d+,\d+,\d+,(true|false)>;',
                   source)
    embed = c['emb'] == 'la'
    if (program is None or lm is None or (program[3] == 'true') != (c['layer_loop'] is True) or
            (program[4] == 'true') != embed or (lm[1] == 'true') != embed or
            int(program[5]) != c['cluster']):
        raise ValueError('native dense program structure differs from its coordinate identity')
    arena, layers = int(program[1]), int(program[2])
    streaming = c['weight_stream'] is True
    # Every typed stage is placed; the ring-less prologue sits in arena 0 and
    # receives no stream.
    if set(placements) != {str(i) for i in range(head + 1)} or placements['0'] != (0, 0):
        raise ValueError('native stage placement differs from coordinate identity')

    def placement(number, typed, stream, full_width=False):
        # D5 arena i % 2 (attention may take the whole budget); D6 streams
        # either the rule's first slots or nothing (prompt §6, §8).
        found = placements.get(number)
        offsets = {0} if full_width else {(int(number) % 2) * arena}
        if (typed != 'Body' or found is None or found[0] not in offsets or
                found[1] not in ((0, stream) if streaming and stream else (0,))):
            raise ValueError('native stage placement differs from coordinate identity')

    # Manifest GEMM i is certified against the typed stage that runs it: the
    # GEMM positions of its layer within the group, or the head. A fused stage
    # runs two manifest GEMMs (expert gate_up, then down).
    positions = [k for k, name in enumerate(layer) if name != 'attention'
                 for _ in range(2 if name == 'expert_fused' else 1)]
    gemms = manifest['gemms']
    if len(gemms) != len(positions)*layers + 1:
        raise ValueError('native dense GEMM call count differs from manifest')
    calls = {number: ('Body' if typed == 'HeadGemm' else typed, kind, raw) for typed, number, kind, raw in
             re.findall(r'using (Stage|Body|HeadGemm)_(\d+)\s*=\s*(Gemm|MoeFused|Moe)<([^;]+)>;', source)}
    expected_calls = {str(1 + (l % per_group)*len(layer) + k) for l in range(per_group) for k in positions}
    if set(calls) != expected_calls | {str(head)}:
        raise ValueError('native dense GEMM call count differs from manifest')
    result = []
    for index, gemm in enumerate(gemms):
        if index == len(positions)*layers:
            number = str(head)
        else:
            l, g = divmod(index, len(positions))
            number = str(1 + (l % per_group)*len(layer) + positions[g])
        typed, kind, raw = calls[number]
        args = raw.split(',')
        family = gemm['family']
        moe = family in MOE_MODES
        if kind != ('MoeFused' if fused and family in FUSED else 'Moe' if moe else 'Gemm'):
            raise ValueError(f'native dense GEMM call differs from manifest at GEMM {index}')
        if kind == 'MoeFused':
            # batch, N_gu, N_dn, K_gu, TN/TK gate_up, TN/TK down, slots, WN/WK
            # gate_up, WN/WK down, U, experts, top-k, float routes, page, 11
            # operands (the 10th gu_done), map, register stages.
            if len(args) != 30 or args[13] != '16' or args[14] not in ('64', '128') or \
                    args[15] != '8' or args[17] not in ('0', '64') or args[27] == 'kAbsent':
                raise ValueError('native fused MoE call signature is invalid')
            down = family == 'expert_down'
            values = list(map(int, args[:13]))
            batch, slots = values[0], values[8]
            n, k = (values[2], values[1] // 2) if down else (values[1], values[3])
            tn, tk = values[6:8] if down else values[4:6]
            wn, wk = values[11:13] if down else values[9:11]
            register_stages = int(args[29])
            stream = 0
        elif moe:
            # batch, N, K, TN, TK, slots, WN, WK, mode, U, experts, top-k,
            # float routes, page, buffers, map, register stages[, softmax-all].
            # 23 arguments: binaries generated before A3 and MR7 (REF-R3, the MR
            # arms) -- 7 buffers, E = 128, softmax over the selected implicit.
            # 25 arguments: 8 buffers (the last is MR7's gu_done, which only a
            # fused plan's router carries) and the router's softmax convention.
            legacy = len(args) == 23
            if len(args) not in (23, 25) or args[8] != 'backend::Sm90DenseMoeMode::' + MOE_MODES[family] or \
                    args[9] != '16' or args[10] not in ('64', '128') or args[11] != '8' or \
                    args[13] not in (('0', '64') if family != 'router' else ('0',)) or \
                    (legacy and (args[10] != '128' or fused)) or \
                    (not legacy and (args[24] not in ('true', 'false') or
                                     (args[21] != 'kAbsent') != (family == 'router' and fused))):
                raise ValueError('native MoE call signature is invalid')
            register_stages = int(args[22 if legacy else 23])
            stream = 0 if family == 'expert_gate_up' else None
        else:
            # ..., op, U, InputNorm, W, C, D, index, register stages, map
            # [, multicast (D10c)].
            multicast = len(args) == 18
            if len(args) not in (17, 18) or (multicast and args[17] != 'true') or \
                    multicast != (c['gemms'][family].get('multicast') is True):
                raise ValueError('native dense GEMM call signature is invalid')
            register_stages = int(args[15])
            if args[10] != ('true' if family in ('qkv', 'gate_up', 'lm_head') else 'false'):
                raise ValueError('native dense GEMM InputNorm differs from its family')
            if args[8] != 'backend::ServingEpilogueOp::' + DENSE_OPS[family]:
                raise ValueError('native dense GEMM epilogue differs from its family')
            if int(args[9]) != c['swiglu_u']:
                raise ValueError(f'native dense GEMM call differs from manifest at GEMM {index}')
            stream = None
        if kind != 'MoeFused':
            batch, n, k, tn, tk, slots, wn, wk = map(int, args[:8])
        actual = dict(tile_m=16, tile_n=tn, tile_k=tk, stages=slots,
                      warps_n=wn, warps_k=wk, split_k=1)
        f = c['gemms'][family]
        placement(number, typed, min(slots, k // tk) * tk * tn * 2 if stream is None else stream)
        if (gemm['index'] != index or gemm.get('impl') != 'native_mma_sync' or
                batch != manifest['runtime_batch'] or gemm['n'] != n or gemm['k'] != k or
                any(gemm[key] != value for key, value in actual.items()) or
                (tn, tk*tn*2, slots, wn, wk) !=
                (f['tile_n'], f['slot_bytes'], f['slots'], f['warps_n'], f['warps_k']) or
                f['mma'] != 'mma_sync' or register_stages != f.get('register_stages', 1) or
                register_stages != gemm.get('register_stages', 1)):
            raise ValueError(f'native dense GEMM call differs from manifest at GEMM {index}')
        row = dict(index=index, compiled_variant=index, impl='native_mma_sync', **actual)
        row['register_stages'] = register_stages
        result.append(row)
    attention = re.findall(r'using (Stage|Body)_(\d+)\s*=\s*Attention<([^;]+)>;', source)
    if not attention:
        raise ValueError('native dense attention call is missing')
    if sorted(number for _, number, _ in attention) != sorted(
            str(1 + l*len(layer) + layer.index('attention')) for l in range(per_group)):
        raise ValueError('native dense attention call differs from manifest')
    a = c['attention']
    expected = [a[k] for k in ('chunks', 'block_keys', 'slots', 'new_row_work')]
    shape = None
    for typed, number, raw in attention:
        args = raw.split(',')
        # batch, Hkv, D, Q, norm, chunks, keys, slots, row work, capacity, 11 buffers, K/V maps
        # [, DSMEM merge (D10b)].
        dsmem = len(args) == 24
        if len(args) not in (23, 24) or (dsmem and args[23] != 'true') or \
                dsmem != (a.get('merge') == 'dsmem'):
            raise ValueError('native dense attention call signature is invalid')
        batch, hkv, dim, queries, norm, chunks, keys, slots, row, capacity = map(int, args[:10])
        current = (dim, queries, norm)
        allowed = current in ATTENTION and hkv == 8 if c['model'] == 'dense' else \
            (hkv, dim, queries, norm) in MOE_ATTENTION
        if (batch != manifest['runtime_batch'] or capacity != manifest['capacity'] or
                [chunks, keys, slots, row] != expected or not allowed or
                (shape is not None and shape != current)):
            raise ValueError('native dense attention call differs from manifest')
        shape = current
        # Sm90DenseLayout::Attention; storage beyond one arena takes the budget.
        merge = 16*queries*dim + 32*queries + 4*max(chunks, 4)*queries + 4 + \
            (4*chunks*queries*dim + 4*chunks*queries if dsmem else 0)
        storage = (4*slots*keys*dim + 96*dim + merge + 1023) // 1024 * 1024
        full = storage > arena
        placement(number, typed, 0 if full else slots*4*keys*dim, full)
    return result


def implementations(manifest):
    c = manifest['native_dense']
    return dict(execution_arch_id=900, program='sm90_dense_native',
                input_norm='consumer_rms', swiglu_interleave_u=c['swiglu_u'],
                gemms=[dict(index=g['index'], impl=g['impl'],
                            task_body='Sm90DenseMoeGemm' if g['family'] in MOE_MODES else 'Sm90DenseGemm')
                       for g in manifest['gemms']],
                attention=dict(impl='native_tma_mma_sync', task_body='Sm90DenseAttentionBody',
                               chunks=c['attention']['chunks'],
                               merge=('dsmem' if c['attention'].get('merge') == 'dsmem' else 'last_arriver')
                               if c['attention']['chunks'] > 1 else 'none',
                               buffers=c['attention']['slots'], partition='actual_past'))
