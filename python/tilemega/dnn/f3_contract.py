"""Certify legacy schema-3 and pipelined schema-4 F3 artifacts."""
import json
import re

BODIES = dict(layout='Sm90F3Layout', pool='Sm90F3MaxPool',
              global_pool='Sm90F3GlobalPool', layer_norm='Sm90F3LayerNorm',
              embedding='Sm90F3EmbeddingSum', encoder_attention='Sm90F3Attention')


def body_of(stage):
    if stage['body'] in ('gemm', 'conv'):
        return 'Sm90F3GemmSwapAB' if stage['mode'] == 'swapab' else 'Sm90F3Gemm'
    if stage['body'] not in BODIES:
        raise ValueError('unsupported F3 body: ' + stage['body'])
    return BODIES[stage['body']]


def repeat_program(manifest, source, program, buffers, maps, bodies):
    if len(manifest['native_forward']['stages']) != 63:
        raise ValueError('F3 Repeat requires the complete twelve-layer BERT program')
    repeated = re.fullmatch(r',Stage_0,Repeat<(6|12)((?:,Loop_\d+)+)>,Stage_61,Stage_62', program[4])
    if repeated is None:
        raise ValueError('F3 Repeat logical stage coverage differs from manifest')
    repetitions = int(repeated[1])
    body = 60//repetitions
    if repeated[2] != ''.join(',Loop_' + str(i) for i in range(body)):
        raise ValueError('F3 Repeat logical stage coverage differs from manifest')
    loops = re.findall(r'using Loop_(\d+)=(RepeatStage<[^;]+>);', source)
    alias_bindings = {int(index): dict(re.findall(r'(k\w+)=(kAbsent|\d+)', fields))
                      for index, fields in re.findall(r'struct LoopBind_(\d+) \{([^}]+)\};', source)}
    if alias_bindings and set(alias_bindings) != set(range(body)):
        raise ValueError('F3 Repeat virtual bindings do not cover its body')
    expected = []
    for i in range(body):
        canonical = 1+body+i
        bind = f'LoopBind_{i}' if alias_bindings else f'Bind_{canonical}'
        normal = f'LoopWrap_{i}' if alias_bindings else f'Wrap_{canonical}'
        first = f'GemmStage<Body_{1+i},{bind},Shape_{1+i}>' if i in (0, 2) else normal
        last = f',GemmStage<Body_60,{bind},Shape_60>' if i+1 == body else ''
        expected.append((str(i), f'RepeatStage<{normal},{first}{last}>'))
    if loops != expected:
        raise ValueError('F3 Repeat edge variants differ from their logical stages')
    shapes = dict(re.findall(r'struct Shape_(\d+) \{([^}]+)\};', source))
    bindings = {}
    for index, fields in re.findall(r'struct Bind_(\d+) \{([^}]+)\};', source):
        bindings[int(index)] = dict(re.findall(r'(k\w+)=(kAbsent|\d+)', fields))
    bf = ('kInput kWeight kOutput kPartial kCounters kVector0 kVector1 kVector2 kVector3 '
          'kResidual kStats kRowStats kRowPartials kRowCounters kChannelPartial kIndex0 kIndex1 kTable').split()
    mf = 'kMap kResidualMap kQueryMap kValueMap kOutputMap'.split()
    if len(bindings) != 63 or any(set(b) != set(bf+mf) for b in bindings.values()):
        raise ValueError('F3 Repeat operand bindings are incomplete')
    loop_wrappers = re.findall(r'using LoopWrap_(\d+)=(\w+)<Body_(\d+),LoopBind_(\d+),Shape_(\d+)>;', source)
    expected_wrappers = []
    if alias_bindings:
        wrappers = dict(re.findall(r'using Wrap_(\d+)=(\w+)<', source))
        next_buffer, next_map = 0, 0
        for i in range(body):
            canonical = 1+body+i
            virtual, physical = alias_bindings[i], bindings[canonical]
            if set(virtual) != set(bf+mf):
                raise ValueError('F3 Repeat virtual operand bindings are incomplete')
            for fields, start in ((bf, next_buffer), (mf, next_map)):
                for field in fields:
                    want = 'kAbsent' if physical[field] == 'kAbsent' else str(start)
                    if virtual[field] != want:
                        raise ValueError('F3 Repeat virtual binding changes an operand role')
                    if want != 'kAbsent':
                        start += 1
                if fields is bf:
                    next_buffer = start
                else:
                    next_map = start
            expected_wrappers.append((str(i), wrappers[str(canonical)], str(canonical), str(i), str(canonical)))
    if loop_wrappers != expected_wrappers:
        raise ValueError('F3 Repeat virtual wrapper differs from its logical stage')
    count, nmap = int(buffers[1]), int(maps[1])
    indices = re.findall(r'std::array<unsigned,(\d+)> indices=\{\{([^}]+)\}\};', source)
    tables = re.findall(r'template<>struct RepeatTables<'+str(repetitions)+r'([^>]+)>', source)
    if (len(indices) != 1 or int(indices[0][0]) != repetitions*(count+nmap)
            or tables != [''.join(',Loop_' + str(i) for i in range(body))]):
        raise ValueError('F3 Repeat binding table dimensions differ from program')
    tokens = [t.strip() for t in indices[0][1].split(',') if t.strip()]
    if any(re.fullmatch(r'kAbsent|\d+u?', t) is None for t in tokens):
        raise ValueError('F3 Repeat remap contains a nonliteral index')
    tokens = [t.removesuffix('u') for t in tokens]
    expected_indices = []
    for rep in range(repetitions):
        row = [str(i) for i in range(count)] + [str(i) for i in range(nmap)]
        assigned = {}
        for i in range(body):
            actual, canonical = 1+body*rep+i, 1+body+i
            edge = (rep == 0 and i in (0, 2)) or (rep+1 == repetitions and i+1 == body)
            if not edge and (bodies[str(actual)][1:] != bodies[str(canonical)][1:]
                             or shapes.get(str(actual)) != shapes.get(str(canonical))):
                raise ValueError('F3 Repeat changes a nonperiodic stage body')
            if actual not in bindings or canonical not in bindings:
                raise ValueError('F3 Repeat stage binding is missing')
            for fields, offset, limit in ((bf, 0, count), (mf, count, nmap)):
                for field in fields:
                    before = (alias_bindings[i] if alias_bindings else bindings[canonical])[field]
                    after = bindings[actual][field]
                    if before == 'kAbsent':
                        if after != 'kAbsent':
                            raise ValueError('F3 Repeat adds an unbound edge operand')
                        continue
                    if int(before) >= limit or (after != 'kAbsent' and int(after) >= limit):
                        raise ValueError('F3 Repeat binding index is out of range')
                    position = offset + int(before)
                    if position in assigned and assigned[position] != after:
                        raise ValueError('F3 Repeat canonical binding aliases conflict')
                    assigned[position] = after
                    row[position] = after
        expected_indices += row
    if tokens != expected_indices:
        raise ValueError('F3 Repeat binding remap differs from logical stage bindings')
    weights = re.findall(r'std::array<RepeatWeight,63> weights=\{\{([^;]+)\}\};', source)
    expected_weights = []
    for i, stage in enumerate(manifest['native_forward']['stages']):
        expected_weights.append(f'{{Bind_{i}::kWeight,unsigned(Wrap_{i}::kWeightBytes)}}'
                                if stage['body'] in ('gemm', 'conv') else '{kAbsent,0}')
    if len(weights) != 1 or re.sub(r'\s+', '', weights[0]) != ','.join(expected_weights) + ',':
        raise ValueError('F3 Repeat weight prefetch table differs from logical stages')


def validate(manifest, defines):
    c = manifest['native_forward']
    if (c.get('schema') not in (3, 4) or c.get('abi') != 'f3' or c.get('model') != 'dnn'
            or c.get('residency') != 1 or manifest.get('residency') != 1
            or manifest.get('phase') != 'forward' or manifest.get('mode') != 'L1'
            or manifest.get('loop') != 1 or manifest.get('runtime_batch', 0) < 1
            or manifest.get('native_origin') not in ('fixed', 'solver')
            or defines.get('TILEMEGA_STATIC_FORWARD_F3') != '1'
            or defines.get('TILEMEGA_ARCH_ID') != '900'
            or defines.get('TILEMEGA_STATIC_FORWARD_NATIVE', '0') != '0'):
        raise ValueError('invalid F3 execution domain or coordinate schema')
    if not c['stages'] or len(c['stages']) != len(manifest.get('stages', [])):
        raise ValueError('F3 manifest and coordinates disagree on stage count')
    contract = manifest.get('register_contract', {})
    threads = 384 if c['schema'] == 4 else 288
    if (contract.get('threads') != threads or contract.get('callee_limit') != 224
            or contract.get('requested_residency') != 1
            or not contract.get('runtime_occupancy_required')):
        raise ValueError('F3 register contract differs from its role ABI')
    for index, stage in enumerate(c['stages']):
        row = manifest['stages'][index]
        if row.get('index') != index or row.get('body') != stage['body'] or row.get('coordinate') != stage:
            raise ValueError('F3 manifest stage differs from coordinate identity')
        body_of(stage)
        if stage['mode'] not in ('normal', 'gather', 'swapab'):
            raise ValueError('unknown F3 operand mode')
        if stage['body'] not in ('gemm', 'conv') and stage['mode'] != 'normal':
            raise ValueError('small F3 stage cannot use a GEMM operand mode')
        if stage['body'] in ('gemm', 'conv'):
            sched = stage.get('sched', 'coop')
            if (sched not in ('coop', 'pp') or (sched == 'pp' and
                    (stage['mode'] != 'normal' or stage['tile_m'] < 128
                     or stage['split_k'] != 1 or stage.get('mc', 0)))):
                raise ValueError('invalid F3 ping-pong coordinate')
            if c['schema'] == 4 and (stage.get('tk') not in (64, 128)
                    or stage.get('store') not in ('reg', 'tma') or stage.get('mc') not in (0, 1)):
                raise ValueError('invalid F3 pipeline coordinates')
            tiles=(32,64,96,128,192,256) if c['schema']==4 else (32,64,128,256)
            if stage['tile_m'] not in (64, 128, 256) or stage['tile_n'] not in tiles:
                raise ValueError('invalid F3 tile')
            if not 2 <= stage['slots'] <= 8 or not 1 <= stage['split_k'] <= (32 if stage['mode'] == 'swapab' else 16):
                raise ValueError('invalid F3 slots or split-K')
            if stage['mode'] == 'gather' and stage['body'] != 'conv':
                raise ValueError('invalid F3 gather coordinate')


def dispatch(manifest, defines, source):
    validate(manifest, defines)
    rows_override = defines.get('TILEMEGA_F3_ATTENTION_ROWS')
    if rows_override is not None and rows_override not in ('16', '32', '64', '128'):
        raise ValueError('invalid explicit F3 attention row count')
    if defines.get('TILEMEGA_F3_SIDE_NORM', '0') not in ('0', '1'):
        raise ValueError('invalid explicit F3 side normalization mode')
    if defines.get('TILEMEGA_F3_REGISTER_BUDGET', '0') not in ('0', '1', '2', '3'):
        raise ValueError('invalid explicit F3 register budget')
    if defines.get('TILEMEGA_F3_CYCLIC_SLOTS', '0') not in ('0', '1'):
        raise ValueError('invalid explicit F3 cyclic slot indexing')
    if defines.get('TILEMEGA_F3_COALESCED_PARTIALS', '0') not in ('0', '1'):
        raise ValueError('invalid explicit F3 partial layout')
    if defines.get('TILEMEGA_F3_MATRIX_STORE', '0') not in ('0', '1'):
        raise ValueError('invalid explicit F3 matrix store')
    if defines.get('TILEMEGA_F3_ATTENTION_VECTOR_STORE', '0') not in ('0', '1'):
        raise ValueError('invalid explicit F3 attention vector store')
    c = manifest['native_forward']
    match = re.search(r'kF3ForwardIdentity\[\]\s*=\s*("(?:\\.|[^"\\])*");', source)
    if match is None or json.loads(json.loads(match[1])) != c:
        raise ValueError('F3 coordinate identity differs from generated source')
    if re.search(r'\b(?:Sm90Forward(?!Op\b)\w+|ForwardGemmStage|ForwardSmallStage|'
                 r'NativeForwardProgram|tilemega_l[12](?:_loop)?_kernel)\b', source):
        raise ValueError('F3 source contains a legacy execution path')
    count = len(c['stages'])
    expected = {str(i) for i in range(count)}
    placements = re.findall(r'using Stage_(\d+)=Placement<Wrap_(\d+),kOffset_(\d+)>;', source)
    wrappers = re.findall(r'using Wrap_(\d+)=(\w+)<Body_(\d+),Bind_(\d+),Shape_(\d+)>;', source)
    bodies = re.findall(r'using Body_(\d+)=(\w+)<([^;]*)>;', source)
    offsets = re.findall(r'constexpr int kOffset_(\d+)=Sm90F3Budget::Offset\((\d+),Wrap_(\d+)::kSharedBytes\);', source)
    for rows in (placements, wrappers, bodies, offsets):
        if len(rows) != count or {row[0] for row in rows} != expected:
            raise ValueError('F3 typed stages do not cover coordinates exactly once')
    if any(len(set(row)) != 1 for row in placements + offsets):
        raise ValueError('F3 arena placement names another stage')
    wrappers = {row[0]: row for row in wrappers}
    bodies = {row[0]: row for row in bodies}
    result = []
    for index, stage in enumerate(c['stages']):
        key = str(index)
        name = body_of(stage)
        wrapper = ('GemmStage' if stage['body'] in ('gemm', 'conv') else
                   'AttentionStage' if stage['body'] == 'encoder_attention' else 'SmallStage')
        if wrappers[key] != (key, wrapper, key, key, key) or bodies[key][1] != name:
            raise ValueError(f'F3 stage {index} runs another typed body')
        args = bodies[key][2].split(',')
        if name == 'Sm90F3Gemm':
            expected_args = [str(stage[k]) for k in ('tile_m', 'tile_n', 'slots', 'split_k')]
            mode = args[4] if len(args)>4 else ''
            allowed = ('Sm90F3Mode::kGather', 'Sm90F3Mode::kTmaStem') if stage['mode']=='gather' and c['schema']==4 else (
                'Sm90F3Mode::kGather' if stage['mode']=='gather' else 'Sm90F3Mode::kNormal',)
            expected_args.append(mode if mode in allowed else '')
            valid = args[:5] == expected_args
            if c['schema'] == 4:
                suffix = []
                while args and args[-1] in ('true','false'):
                    suffix.insert(0,args.pop()=='true')
                valid = (valid and len(suffix)<=3 and args[-1] == str(stage['tk'])
                         and (suffix[0] if suffix else False) == (stage['store']=='tma')
                         and (suffix[1] if len(suffix)>=2 else False) == bool(stage['mc'])
                         and (suffix[2] if len(suffix)==3 else False) == (stage.get('sched','coop')=='pp'))
        elif name == 'Sm90F3GemmSwapAB':
            valid = (args[:2] == [str(stage[k]) for k in ('slots', 'split_k')]
                     and 1 <= int(args[2]) <= 16 and stage['tile_n'] == 128)
            if c['schema'] == 4:
                valid = valid and args[-1] == str(stage['tk'])
        elif name == 'Sm90F3Layout':
            valid = len(args) == 2 and int(args[0]) > 0 and int(args[1]) == manifest['runtime_batch']
        elif name == 'Sm90F3MaxPool':
            valid = (len(args) in ((6,8) if c['schema']==4 else (6,)) and args[:4] in
                     (['3', '3', '2', '1'], ['2', '2', '2', '0'])
                     and int(args[4]) > 0 and int(args[4]) % 8 == 0
                     and int(args[5]) == manifest['runtime_batch']
                     and (len(args)==6 or all(int(a)>0 for a in args[6:])))
        elif name == 'Sm90F3GlobalPool':
            valid = len(args) == 1 and int(args[0]) == manifest['runtime_batch']
        elif name == 'Sm90F3LayerNorm':
            valid = len(args) == 2 and int(args[0]) > 0 and int(args[1]) > 0
        elif name == 'Sm90F3Attention':
            batch=manifest['runtime_batch']
            valid = args == ([str(batch),rows_override or str(min(128,16*batch))] if c['schema']==4
                             else [str(batch),'128','64','12'])
        else:
            valid = len(args) == 1 and int(args[0]) > 0
        if not valid:
            raise ValueError(f'F3 stage {index} differs from its coordinate or batch')
        result.append(dict(index=index, body=stage['body'], task_body=name, stage_wrapper=wrapper))
    program = re.search(r'using F3ForwardProgram=Program<(\d+),(\d+),(\d+)([^;]+)>;', source)
    buffers = re.search(r'std::array<tm_buffer_info,(\d+)> kF3ForwardBuffers=', source)
    maps = re.search(r'std::array<ForwardMapSpec,(\d+)> kF3ForwardMaps=', source)
    if (program is None or int(program[1]) != manifest['runtime_batch']
            or buffers is None or int(program[2]) != int(buffers[1])
            or int(program[2]) < len(manifest['buffers'])
            or maps is None or int(program[3]) != int(maps[1])):
        raise ValueError('F3 program structure differs from manifest')
    if 'Repeat<' in program[4]:
        repeat_program(manifest, source, program, buffers, maps, bodies)
    elif (program[4] != ''.join(',Stage_' + str(i) for i in range(count))
          or re.search(r'\b(?:Repeat|RepeatTables|RepeatStage)<', source)):
        raise ValueError('F3 program stage coverage differs from manifest')
    return result


def implementations(manifest):
    c = manifest['native_forward']
    return dict(execution_arch_id=900, program='sm90_forward_f3', residency=1,
                stages=[dict(index=i, body=s['body'], task_body=body_of(s))
                        for i, s in enumerate(c['stages'])])
