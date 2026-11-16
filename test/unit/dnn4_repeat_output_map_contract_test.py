#!/usr/bin/env python3
"""Check optional output-map roles against every physical Repeat binding."""
import argparse
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python'))
from tilemega.build.feature_contract import build_defines
from tilemega.dnn.f3_contract import dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--flat', type=Path, required=True)
    args = parser.parse_args()
    source = Path(str(args.artifact)+'.cu').read_text()
    manifest = json.loads(Path(str(args.artifact)+'.plan.json').read_text())
    flat = Path(str(args.flat)+'.cu').read_text()
    assert len(dispatch(manifest, build_defines(source, []), source)) == 63
    assert len(dispatch(manifest, build_defines(flat, []), flat)) == 63
    for prefix in ('Bind_', 'Shape_'):
        parse = lambda text: dict(re.findall(r'struct '+prefix+r'(\d+) \{(.*?)\n\};', text, re.S))
        assert parse(source) == parse(flat)
    rejected = []
    def reject(name, damaged):
        try:
            dispatch(manifest, build_defines(damaged, []), damaged)
        except ValueError:
            rejected.append(name)
            return
        raise AssertionError('accepted corrupted '+name)
    flag = '#define TILEMEGA_F3_REPEAT_OUTPUT_MAP 1\n'
    assert source.count(flag) == 1
    reject('missing flag', source.replace(flag, ''))
    reject('invalid flag', source.replace(flag, flag.replace(' 1', ' 2')))
    optional = re.search(r'(struct LoopBind_0 \{.*?kOutputMap=)(\d+)u', source, re.S)
    assert optional
    for name, value in [('missing optional map', 'kAbsent'), ('aliased optional map', '0u'), ('out of range map', '9999u')]:
        reject(name, source[:optional.start(2)]+value+source[optional.end(2)+1:])
    count, nmap = map(int, re.search(r'using F3ForwardProgram=Program<16,(\d+),(\d+),', source).groups())
    stride = count+nmap
    indices = re.search(r'(std::array<unsigned,\d+> indices=\{\{)(.*?)(\n  \}\};)', source, re.S)
    tokens = list(re.finditer(r'kAbsent|\d+u', indices[2]))
    assert len(tokens) == 12*stride
    for rep in range(12):
        token = tokens[rep*stride+count+int(optional[2])]
        replacement = 'kAbsent' if rep == 0 else '1u'
        reject('wrong optional map '+str(rep), source[:indices.start(2)+token.start()]+replacement+source[indices.start(2)+token.end():])
    binding = re.search(r'struct LoopBind_0 \{.*?\n\};', source, re.S)
    for field in ('kInput', 'kWeight', 'kOutput', 'kMap', 'kVector0', 'kStats'):
        match = re.search(field+r'=(kAbsent|\d+u)', binding[0])
        assert match
        value = '0u' if match[1] == 'kAbsent' else str(int(match[1][:-1])+1)+'u'
        start, end = binding.start()+match.start(1), binding.start()+match.end(1)
        reject('unrelated role '+field, source[:start]+value+source[end:])
    for name, before, after in [
            ('missing wrapper', 'using LoopWrap_0=', 'using MissingWrap_0='),
            ('weight prefetch', '{Bind_1::kWeight,unsigned(Wrap_1::kWeightBytes)}', '{Bind_6::kWeight,unsigned(Wrap_1::kWeightBytes)}')]:
        assert before in source
        reject(name, source.replace(before, after, 1))
    print(json.dumps(dict(passed=True, evidence='verified CPU', logical_stages=63,
                          corruptions_rejected=len(rejected))))


if __name__ == '__main__':
    main()
