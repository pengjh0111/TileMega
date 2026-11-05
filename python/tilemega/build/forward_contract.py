"""Certify native DNN forward coordinates against generated, typed stages."""
import json
import re

# One typed body and one stage wrapper per coordinate body (prompt §10 N-1).
BODIES = {'gemm': 'Sm90ForwardGemm', 'conv': 'Sm90ForwardGemm',
          'depthwise': 'Sm90ForwardDepthwise', 'dwpw': 'Sm90ForwardDwPw',
          'inverted_residual': 'Sm90ForwardDwPw',
          'encoder_attention': 'Sm90ForwardEncoderAttention',
          'layout': 'Sm90ForwardLayout', 'pool': 'Sm90ForwardMaxPool',
          'global_pool': 'Sm90ForwardGlobalPool', 'gemv': 'Sm90ForwardGemv',
          'layer_norm': 'Sm90ForwardLayerNorm', 'embedding': 'Sm90ForwardEmbeddingSum'}
WRAPPERS = {'gemm': 'ForwardGemmStage', 'conv': 'ForwardGemmStage',
            'depthwise': 'ForwardDepthwiseStage', 'dwpw': 'ForwardFusedStage',
            'inverted_residual': 'ForwardFusedStage',
            'encoder_attention': 'ForwardAttentionStage'}
LEGACY = (r'\b(?:GemmInvocation|StageDesc|ServingGemmTaskBody|ServingGemvTaskBody|'
          r'ConvTaskBody|DepthwiseConvTaskBody|EncoderAttentionTaskBody|'
          r'tilemega_l1_kernel|tilemega_l1_loop_kernel|tilemega_l2_kernel)\b')


def validate(manifest, defines):
    c = manifest['native_forward']
    required = dict(phase='forward', mode='L1', loop=1)
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('native forward execution domain differs from its manifest')
    if (defines.get('TILEMEGA_STATIC_FORWARD_NATIVE') != '1' or
            defines.get('TILEMEGA_ARCH_ID') != '900' or
            manifest.get('native_origin') not in ('fixed', 'solver') or
            c.get('schema') not in (1, 2) or c.get('model') != 'dnn' or
            manifest.get('residency') != c['residency'] or
            manifest.get('runtime_batch', 0) < 1):
        raise ValueError('native forward architecture or coordinate schema is invalid')
    if len(c['stages']) != len(manifest.get('stages', [])):
        raise ValueError('native forward manifest and coordinates disagree on the stage count')


def head(raw, count):
    """The first `count` template arguments, which carry no nested template."""
    args = raw.split(',')
    if len(args) < count:
        raise ValueError('native forward typed stage has too few arguments')
    return args[:count]


def dispatch(manifest, defines, source):
    validate(manifest, defines)
    c = manifest['native_forward']
    match = re.search(r'kNativeForwardIdentity\[\]\s*=\s*("(?:\\.|[^"\\])*");', source)
    if match is None or json.loads(json.loads(match[1])) != c:
        raise ValueError('native forward coordinate identity differs from generated source')
    if re.search(LEGACY, source):
        raise ValueError('native forward generated source contains a legacy execution path')
    arena = re.search(r'constexpr int kNativeForwardArena=(\d+);', source)
    if arena is None or int(arena[1]) <= 0 or int(arena[1]) % 1024:
        raise ValueError('native forward arena budget is invalid')
    schema = c['schema']
    stages = len(c['stages'])
    placements = dict(re.findall(r'using Stage_(\d+)=Placement<Wrap_(\d+),kOffset_\1>;', source))
    wrappers = dict(re.findall(r'using Wrap_(\d+)=(\w+)<Body_\1,Bind_\1,Shape_\1>;', source))
    bodies = {number: name for number, name in
              re.findall(r'using Body_(\d+)=(\w+)(?:<[^;]*>)?;', source)}
    if (set(placements) != {str(i) for i in range(stages)} or
            any(placements[k] != k for k in placements) or
            set(wrappers) != set(placements) or set(bodies) != set(placements)):
        raise ValueError('native forward typed stages do not cover the coordinates')
    result = []
    for index, stage in enumerate(c['stages']):
        key = str(index)
        body = stage['body']
        if body not in BODIES:
            raise ValueError('unknown native forward body: ' + body)
        if bodies[key] != BODIES[body] or wrappers[key] != WRAPPERS.get(body, 'ForwardSmallStage'):
            raise ValueError(f'native forward stage {index} runs another typed body')
        raw = re.search(rf'using Body_{index}={BODIES[body]}<([^;]*)>;', source)
        arguments = raw[1] if raw else ''
        if body in ('gemm', 'conv'):
            tm, tn, tk, slots, split, mode = head(arguments, 6)
            if (int(tm) != (64 if schema == 1 else stage['tile_m']) or
                    int(tm) not in (32, 64, 128) or tk != '64' or int(tn) != stage['tile_n'] or
                    int(slots) != stage['slots'] or int(split) != stage['split_k'] or
                    mode != 'Sm90ForwardAMode::' + ('kIm2col' if body == 'conv' else 'kDense')):
                raise ValueError(f'native forward GEMM stage {index} differs from its coordinate')
        elif body == 'depthwise':
            tw, th, block, _stride, slots = head(arguments, 5)
            if (int(tw) != stage['tile_w'] or int(th) != stage['tile_h'] or
                    int(block) != stage['channel_block'] or int(slots) != stage['slots']):
                raise ValueError(f'native forward depthwise stage {index} differs from its coordinate')
        elif body in ('dwpw', 'inverted_residual'):
            tw, th, block, tn, _stride, slots = head(arguments, 6)
            if (int(tw) != stage['tile_w'] or int(th) != stage['tile_h'] or
                    int(block) != stage['channel_block'] or int(tn) != stage['tile_n'] or
                    int(slots) != stage['slots'] or not stage['fused']):
                raise ValueError(f'native forward fused stage {index} differs from its coordinate')
        elif body == 'encoder_attention':
            if schema == 1:
                sequence, dim, _heads, rows, slots = head(arguments, 5)
                valid = dim == '64' and rows == '16' and int(slots) == stage['slots'] and int(sequence) % 64 == 0
            else:
                batch, sequence, dim, heads, rows, slots = head(arguments, 6)
                valid = (int(batch) == manifest['runtime_batch'] and dim == '64' and int(heads) > 0
                         and int(rows) == stage['tile_m'] and int(rows) in (16,32,64,128)
                         and int(sequence) % 64 == 0 and int(sequence) % int(rows) == 0
                         and int(slots) == stage['slots'])
                if len(arguments.split(',')) > 6:
                    raise ValueError('production attention cannot select a scalar reference')
            if not valid:
                raise ValueError(f'native forward attention stage {index} differs from its coordinate')
        elif schema == 2 and body == 'gemv':
            block, contraction = head(arguments, 2)
            if int(block) != stage['tile_n'] or int(block) not in (64,128,256) or int(contraction) <= 0 or int(contraction) % 64:
                raise ValueError(f'native forward GEMV stage {index} differs from its coordinate')
        elif schema == 2 and body == 'layout':
            channels, pixels, batch = head(arguments, 3)
            if int(channels) < 1 or int(pixels) != 128 or int(batch) != manifest['runtime_batch']:
                raise ValueError(f'native forward layout stage {index} differs from its batch')
        elif schema == 2 and body == 'pool':
            window, stride, batch = head(arguments, 3)
            if window != '3' or stride != '2' or int(batch) != manifest['runtime_batch']:
                raise ValueError(f'native forward pool stage {index} differs from its batch')
        elif schema == 2 and body == 'global_pool':
            if int(arguments) != manifest['runtime_batch']:
                raise ValueError(f'native forward global pool stage {index} differs from its batch')
        if schema == 2 and body in ('depthwise','dwpw','inverted_residual'):
            if int(arguments.rsplit(',',1)[1]) != manifest['runtime_batch']:
                raise ValueError(f'native forward spatial stage {index} differs from its batch')
        result.append(dict(index=index, body=body, task_body=BODIES[body],
                           stage_wrapper=wrappers[key]))
    program = re.search(r'using NativeForwardProgram=Program<(\d+),(\d+),(\d+),(\d+),'
                        r'kNativeForwardArena((?:,Stage_\d+)+)>;', source)
    if program is None:
        raise ValueError('native forward program structure is missing')
    listed = re.findall(r'Stage_(\d+)', program[5])
    if (int(program[1]) != manifest['runtime_batch'] or int(program[2]) != c['residency'] or
            listed != [str(i) for i in range(stages)]):
        raise ValueError('native forward program structure differs from its coordinate identity')
    return result


def implementations(manifest):
    c = manifest['native_forward']
    return dict(execution_arch_id=900, program='sm90_forward_native',
                residency=c['residency'],
                stages=[dict(index=i, body=s['body'], task_body=BODIES[s['body']])
                        for i, s in enumerate(c['stages'])])
