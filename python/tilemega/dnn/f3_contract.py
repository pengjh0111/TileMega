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
            expected_args.append('Sm90F3Mode::' + ('kGather' if stage['mode'] == 'gather' else 'kNormal'))
            valid = args[:5] == expected_args
            if c['schema'] == 4:
                valid = valid and args[-1] == str(stage['tk'])
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
            valid = args == ([str(batch),str(min(128,16*batch))] if c['schema']==4
                             else [str(batch),'128','64','12'])
        else:
            valid = len(args) == 1 and int(args[0]) > 0
        if not valid:
            raise ValueError(f'F3 stage {index} differs from its coordinate or batch')
        result.append(dict(index=index, body=stage['body'], task_body=name, stage_wrapper=wrapper))
    program = re.search(r'using F3ForwardProgram=Program<(\d+),(\d+),(\d+)((?:,Stage_\d+)+)>;', source)
    buffers = re.search(r'std::array<tm_buffer_info,(\d+)> kF3ForwardBuffers=', source)
    maps = re.search(r'std::array<ForwardMapSpec,(\d+)> kF3ForwardMaps=', source)
    if (program is None or int(program[1]) != manifest['runtime_batch']
            or buffers is None or int(program[2]) != int(buffers[1])
            or int(program[2]) < len(manifest['buffers'])
            or maps is None or int(program[3]) != int(maps[1])
            or re.findall(r'Stage_(\d+)', program[4]) != [str(i) for i in range(count)]):
        raise ValueError('F3 program structure differs from manifest')
    return result


def implementations(manifest):
    c = manifest['native_forward']
    return dict(execution_arch_id=900, program='sm90_forward_f3', residency=1,
                stages=[dict(index=i, body=s['body'], task_body=body_of(s))
                        for i, s in enumerate(c['stages'])])
