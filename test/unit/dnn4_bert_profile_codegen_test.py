#!/usr/bin/env python3
"""Check explicit BERT mechanism profiles and their query tensor-map boxes."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python'))
from tilemega.build.feature_contract import build_defines
from tilemega.dnn.f3_contract import dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'bridge', 'coordinates', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    environment = {k: v for k, v in os.environ.items() if not k.startswith('TILEMEGA_F3_')}
    environment['CUDA_VISIBLE_DEVICES'] = ''

    def generate(name, settings, succeeds=True):
        source = args.out/(name+'.cu')
        command = [str(args.binary.resolve()), 'compile', str(args.bridge.resolve()),
                   str(source.resolve()), '--emit', 'serving', '--serving', 'forward',
                   '--frontend', 'dnn', '--batch', '16', '--abi', 'f3',
                   '--tmexec-config', str(args.coordinates.resolve()),
                   '--runtime-target', str(ROOT/'configs/targets/sm_90_h800.json')]
        with source.with_suffix('.log').open('w') as log:
            done = subprocess.run(command, env=dict(environment, **settings),
                                  stdout=log, stderr=subprocess.STDOUT)
        assert (done.returncode == 0) == succeeds
        if not succeeds:
            return
        text = source.read_text()
        manifest = json.loads(Path(str(source)+'.plan.json').read_text())
        assert len(dispatch(manifest, build_defines(text, []), text)) == 63
        return text, manifest

    flat, _ = generate('default', {})
    side, _ = generate('side', {'TILEMEGA_F3_SIDE_NORM': '1'})
    assert side == '#define TILEMEGA_F3_SIDE_NORM 1\n'+flat
    budget, _ = generate('register64', {'TILEMEGA_F3_REGISTER_BUDGET': '1'})
    assert budget == '#define TILEMEGA_F3_REGISTER_BUDGET 1\n'+flat
    budget, _ = generate('register72', {'TILEMEGA_F3_REGISTER_BUDGET': '2'})
    assert budget == '#define TILEMEGA_F3_REGISTER_BUDGET 2\n'+flat
    rows, manifest = generate('rows64', {'TILEMEGA_F3_SIDE_NORM': '1',
                                       'TILEMEGA_F3_ATTENTION_ROWS': '64'})
    query_maps = [int(x) for x in re.findall(r'kQueryMap=(\d+)', side)]
    assert len(query_maps) == len(set(query_maps)) == 12
    map_lines = side.split(' kF3ForwardMaps={{\n', 1)[1].split('\n}};', 1)[0].splitlines()
    expected = side.replace('Sm90F3Attention<16,128>', 'Sm90F3Attention<16,64>')
    for index in query_maps:
        before = map_lines[index]
        after = before.replace('{64u,1u,128u,0u}', '{64u,1u,64u,0u}')
        assert after != before and side.count(before) == 1
        expected = expected.replace(before, after)
    expected = expected.replace('#define TILEMEGA_F3_SIDE_NORM 1\n',
        '#define TILEMEGA_F3_SIDE_NORM 1\n#define TILEMEGA_F3_ATTENTION_ROWS 64\n', 1)
    assert rows == expected
    for bad in (rows.replace('Sm90F3Attention<16,64>', 'Sm90F3Attention<16,128>'),
                rows.replace('#define TILEMEGA_F3_ATTENTION_ROWS 64\n', '')):
        try:
            dispatch(manifest, build_defines(bad, []), bad)
        except ValueError:
            pass
        else:
            raise AssertionError('unidentified attention row change accepted')
    generate('invalid_rows', {'TILEMEGA_F3_ATTENTION_ROWS': '48'}, succeeds=False)
    generate('invalid_side', {'TILEMEGA_F3_SIDE_NORM': '2'}, succeeds=False)
    generate('invalid_budget', {'TILEMEGA_F3_REGISTER_BUDGET': '3'}, succeeds=False)
    print(json.dumps(dict(status='passed', evidence='verified', profiles=5,
                          query_maps=12, invalid_settings_rejected=3)))


if __name__ == '__main__':
    main()
