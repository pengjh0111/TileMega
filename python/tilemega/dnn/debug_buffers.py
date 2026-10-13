"""Find materialized DNN outputs in the identity-bound generated tables.

Deferred normalization retains unused buffer descriptors. Allocation alone is
not evidence that a buffer contains an intermediate activation. This parser is
deliberately limited to generated aggregate initializers and fails closed on
unknown task kinds; it never evaluates C++ expressions.
"""
import re


def fields(text):
    parts, begin, depth = [], 0, 0
    for index, char in enumerate(text):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth < 0:
                raise ValueError('unbalanced generated initializer')
        elif char == ',' and depth == 0:
            if text[begin:index].strip():
                parts.append(text[begin:index].strip())
            begin = index + 1
    if depth:
        raise ValueError('unbalanced generated initializer')
    if text[begin:].strip():
        parts.append(text[begin:].strip())
    return parts


def aggregate(text):
    if not text.startswith('{') or not text.endswith('}'):
        raise ValueError('expected generated aggregate initializer')
    return fields(text[1:-1])


def table(source, kind, name):
    match = re.search(r'constexpr\s+' + kind + r'\s+' + name + r'\[\] = \{(.*?)\n\};', source, re.S)
    if not match:
        raise ValueError('missing generated table: ' + name)
    return [aggregate(row) for row in fields(match[1])]


def integer(text):
    if text in ('kNoOperand', 'kDmNoIndex'):
        return 0xffffffff
    if not re.fullmatch(r'(?:0x[0-9a-fA-F]+|[0-9]+)u?', text):
        raise ValueError('unsupported generated buffer index: ' + text)
    return int(text.rstrip('u'), 16 if text.startswith('0x') else 10)


def materialized_buffers(source, manifest):
    """Return written/internal and externally initialized buffer names.

    Operand positions follow DmStageRunner and GemmDesc. Side outputs are
    emitted explicitly by EmitDm; their buffer and auxiliary are both kept.
    """
    gemms = table(source, 'GemmDesc', 'kGemms')
    stages = table(source, 'StageDesc', 'kStages')
    buffers = manifest['buffers']
    writers = {}

    def mark(index, stage):
        if index == 0xffffffff:
            return
        if index >= len(buffers):
            raise ValueError('generated output index exceeds buffer table')
        writers.setdefault(index, []).append(stage)

    outputs = {'kLayoutConvert': (1,), 'kLayerNorm': (3, 4),
               'kDepthwiseConv': (2, 3), 'kPool': (1,),
               'kGlobalPoolReduce': (1, 2), 'kEmbeddingSum': (5, 6),
               'kEncoderAttention': (1,)}
    for number, stage in enumerate(stages):
        kind = stage[0].removeprefix('TaskKind::')
        if kind in ('kGemm', 'kDwPwFused'):
            gemm = gemms[integer(stage[1])]
            mark(integer(gemm[5]), number)
            descriptors = ','.join(gemm)
        elif kind in outputs:
            operands = aggregate(stage[5])
            for position in outputs[kind]:
                mark(integer(operands[position]), number)
            descriptors = ','.join(stage)
        else:
            raise ValueError('unknown diagnostic output contract: ' + kind)
        for match in re.finditer(r'\{(?:DmSideOutputKind::\w+|static_cast<DmSideOutputKind>\([0-9]+u\)),\s*([^,{}]+),\s*([^,{}]+),', descriptors):
            mark(integer(match[1].strip()), number)
            mark(integer(match[2].strip()), number)
    return {buffer['name']: dict(writers=writers.get(index, []),
                                external=buffer.get('role') == 'external')
            for index, buffer in enumerate(buffers)
            if index in writers or buffer.get('role') == 'external'}
