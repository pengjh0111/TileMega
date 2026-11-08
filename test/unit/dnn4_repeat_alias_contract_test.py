#!/usr/bin/env python3
"""Certify per-role virtual IDs and reject corrupted alias remaps."""
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
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--original-flat', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    manifest = json.loads(Path(str(args.source)+'.plan.json').read_text())
    defines = build_defines(source, [])
    assert len(dispatch(manifest, defines, source)) == 63
    assert args.source.with_name('repeat0.cu').read_text() == args.original_flat.read_text()
    body = 60//int(re.search(r',Stage_0,Repeat<(6|12),', source)[1])
    assert len(re.findall(r'struct LoopBind_\d+ ', source)) == body
    changes = []
    for name, before, after in [
        ('missing virtual stage', 'struct LoopBind_0 ', 'struct MissingBind_0 '),
        ('virtual weight alias', 'kInput=kAbsent,kWeight=0u,kOutput=1u',
         'kInput=kAbsent,kWeight=1u,kOutput=1u'),
        ('virtual map alias', 'struct LoopBind_0 {\n  static constexpr unsigned kMap=0u',
         'struct LoopBind_0 {\n  static constexpr unsigned kMap=1u'),
        ('virtual wrapper', f'LoopWrap_0=GemmStage<Body_{1+body},LoopBind_0,Shape_{1+body}>',
         f'LoopWrap_0=GemmStage<Body_{2+body},LoopBind_0,Shape_{1+body}>'),
        ('physical weight prefetch', '{Bind_1::kWeight,unsigned(Wrap_1::kWeightBytes)}',
         '{LoopBind_0::kWeight,unsigned(Wrap_1::kWeightBytes)}')]:
        assert before in source, name
        changes.append((name, source.replace(before, after, 1)))
    table = re.search(r'(std::array<unsigned,\d+> indices=\{\{\s*)(\d+)u', source)
    changes.append(('remapped virtual input', source[:table.start(2)]+str(int(table[2])+1)+source[table.end(2):]))
    for name, damaged in changes:
        try:
            dispatch(manifest, build_defines(damaged, []), damaged)
        except ValueError:
            continue
        raise AssertionError('accepted corrupted '+name)
    print(json.dumps(dict(status='passed', evidence='verified', logical_stages=63,
                          virtual_stages=body, corruptions_rejected=len(changes),
                          original_flat_source_unchanged=True)))


if __name__ == '__main__':
    main()
