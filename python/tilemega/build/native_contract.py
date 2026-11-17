"""Certify native dense coordinates against generated, typed call sites."""
import json
import re


# Typed stages of one layer (prompt §6 D7, §8); a group holds an even number.
LAYER = dict(dense=('qkv', 'attention', 'o', 'gate_up', 'down'),
             moe=('qkv', 'attention', 'o', 'router', 'expert_gate_up', 'expert_down'))
DENSE_OPS = dict(qkv='kStore', o='kResidual', gate_up='kSwiGLU', down='kResidual',
                 lm_head='kArgmaxPartial')
MOE_MODES = dict(router='kRouter', expert_gate_up='kGateUp', expert_down='kDown')
ATTENTION = ((128, 2, 1), (64, 4, 0), (128, 3, 0), (128, 4, 0))
MOE_ATTENTION = ((4, 128, 8, 1),)


def validate(manifest, defines):
    c = manifest['native_dense']
    required = dict(phase='decode', pg='native', mode='L1', loop=1, capacity=1088)
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('native dense execution domain differs from its manifest')
    if (defines.get('TILEMEGA_STATIC_DENSE_NATIVE') != '1' or
            defines.get('TILEMEGA_ARCH_ID') != '900' or
            manifest.get('runtime_batch') not in (1, 2, 4, 8, 16) or
            manifest.get('native_origin') not in ('fixed', 'solver') or c.get('schema') != 5 or
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
    if manifest.get('realization') == 'explicit_stages_qwen_b1_v1':
        return explicit_dispatch(manifest, source)
    layer = LAYER[c['model']]
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
    # GEMM positions of its layer within the group, or the head.
    positions = [k for k, name in enumerate(layer) if name != 'attention']
    gemms = manifest['gemms']
    if len(gemms) != len(positions)*layers + 1:
        raise ValueError('native dense GEMM call count differs from manifest')
    calls = {number: ('Body' if typed == 'HeadGemm' else typed, kind, raw) for typed, number, kind, raw in
             re.findall(r'using (Stage|Body|HeadGemm)_(\d+)\s*=\s*(Gemm|Moe)<([^;]+)>;', source)}
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
        if kind != ('Moe' if moe else 'Gemm'):
            raise ValueError(f'native dense GEMM call differs from manifest at GEMM {index}')
        if moe:
            # batch, N, K, TN, TK, slots, WN, WK, mode, U, experts, top-k,
            # float routes, page, 7 buffers, map, register stages.
            if len(args) != 23 or args[8] != 'backend::Sm90DenseMoeMode::' + MOE_MODES[family] or \
                    args[9] != '16' or args[11] != '8' or \
                    args[13] not in (('0', '64') if family != 'router' else ('0',)):
                raise ValueError('native MoE call signature is invalid')
            register_stages = int(args[22])
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
    explicit = manifest.get('realization') == 'explicit_stages_qwen_b1_v1'
    result = dict(execution_arch_id=900, program='sm90_dense_native',
                input_norm='consumer_rms', swiglu_interleave_u=c['swiglu_u'],
                gemms=[dict(index=g['index'], impl=g['impl'],
                            task_body='Sm90DenseMoeGemm' if g['family'] in MOE_MODES else 'Sm90DenseGemm')
                       for g in manifest['gemms']],
                attention=dict(impl='native_tma_mma_sync', task_body='Sm90DenseAttentionBody',
                               chunks=c['attention']['chunks'],
                               merge=('dsmem' if c['attention'].get('merge') == 'dsmem' else 'last_arriver')
                               if c['attention']['chunks'] > 1 else 'none',
                               buffers=c['attention']['slots'], partition='actual_past'))

    if explicit:
        result.update(program='sm90_explicit_stages_qwen_b1', input_norm='explicit_rms_statistics',
            residual='explicit_output', swiglu='explicit_output', argmax='explicit_final_reduction',
            embedding='explicit_stage', attention_prepare='explicit_qk_norm_rope_kv_append')
        result['attention']['merge'] = 'explicit_stage' if c['attention']['chunks'] > 1 else 'none'
    return result


def explicit_dispatch(manifest, source):
    c = manifest['native_dense']
    if (manifest['runtime_batch'] != 1 or c['model'] != 'dense' or c['emb'] != 'stage' or
            c['group_wait'] or c['weight_stream'] or c['cluster'] != 1 or c['residency'] != 1):
        raise ValueError('explicit stage realization requires its fixed basic domain')
    chunks = c['attention']['chunks']
    if chunks not in (1,16):raise ValueError('explicit realization supports S1/S16 only')
    names = ['RmsQkv','Qkv','PrepareQkv','AttentionCompute']
    if chunks == 16:names.append('MergeAttention')
    names += ['OutputGemm','OutputResidual','RmsGate','GateUp','SwiGLU','Down','DownResidual']
    if f"using ExplicitLayer=std::tuple<{','.join(names)}>;" not in source:
        raise ValueError('explicit semantic stage list differs')
    if 'using ExplicitEnd=std::tuple<RmsHead,HeadPartial,ArgmaxFinal>;' not in source:
        raise ValueError('explicit head completion is missing')
    if re.search(r'=Head<|=Program<',source):raise ValueError('fused P1 program in P0')
    if not re.search(r'using NativeProgram=ExplicitStageProgram<\d+,\d+,28,\d+,217664,', source):
        raise ValueError('explicit program shape or shared budget differs')
    sequence = manifest.get('execution') == 'kernel_sequence'
    if ('using NativeSequence=ExplicitSequence<NativeProgram>' in source) != sequence:
        raise ValueError('explicit graph uses a different stage list')
    aliases = dict(re.findall(r'using (\w+)Body=Gemm<([^;]+)>;',source))
    families = dict(qkv='Qkv',o='OutputGemm',gate_up='GateUp',down='Down',lm_head='HeadPartial')
    if set(aliases) != set(families.values()):raise ValueError('explicit GEMM family count differs')
    result=[]
    for index,g in enumerate(manifest['gemms']):
        f=g['family'];a=aliases[families[f]].split(',');cfg=c['gemms'][f]
        normalized=f in ('qkv','gate_up','lm_head')
        if len(a)!=19 or a[17]!='false' or a[18]!=str(normalized).lower():
            raise ValueError('explicit GEMM must consume external statistics where required')
        batch,n,k,tn,tk,slots,wn,wk=map(int,a[:8]);rs=int(a[15])
        actual=dict(tile_m=16,tile_n=tn,tile_k=tk,stages=slots,warps_n=wn,warps_k=wk,split_k=1)
        if (batch!=1 or (n,k)!=(g['n'],g['k']) or a[10]!=str(normalized).lower() or
                a[8]!='backend::ServingEpilogueOp::'+('kArgmaxPartial' if f=='lm_head' else 'kStore') or
                int(a[9])!=c['swiglu_u'] or any(g[key]!=v for key,v in actual.items()) or
                (tn,2*tn*tk,slots,wn,wk,rs)!=(cfg['tile_n'],cfg['slot_bytes'],cfg['slots'],
                    cfg['warps_n'],cfg['warps_k'],cfg['register_stages'])):
            raise ValueError('explicit GEMM geometry/realization differs from manifest')
        result.append(dict(index=index,compiled_variant=index,impl='native_mma_sync',register_stages=rs,**actual))
    attention = re.search(r'using AttentionPrepared=Attention<([^;]+)>;',source)
    original = re.search(r'using AttentionOriginal=Attention<([^;]+)>;',source)
    if not attention or not original or attention[1]!=original[1]+',false,true':
        raise ValueError('explicit attention must retain original geometry and disable fused preparation/merge')
    a=attention[1].split(',')
    if list(map(int,a[:10]))!=[1,8,128,2,1,chunks,c['attention']['block_keys'],
            c['attention']['slots'],c['attention']['new_row_work'],1088]:
        raise ValueError('explicit attention shape differs')
    return result
