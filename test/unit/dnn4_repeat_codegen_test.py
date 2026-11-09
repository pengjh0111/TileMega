#!/usr/bin/env python3
"""Check real BERT codegen coverage and exact flat fallback for Repeat."""
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
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--layers', choices=('1', '2'), required=True)
    parser.add_argument('--expect', choices=('repeat', 'flat'), required=True)
    parser.add_argument('--register-budget', choices=('0', '1', '2', '3'),
                        default=os.environ.get('TILEMEGA_F3_REGISTER_BUDGET', '0'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = []
    for setting, arithmetic in (('0', '0'), ('1', '0'), ('1', '1'), ('1', '2')):
        source = args.out/f'repeat{setting}-arithmetic{arithmetic}.cu'
        command = [str(args.binary.resolve()), 'compile', str(args.bridge.resolve()),
                   str(source.resolve()), '--emit', 'serving', '--serving', 'forward',
                   '--frontend', 'dnn', '--batch', str(args.batch), '--abi', 'f3',
                   '--tmexec-config', str(args.coordinates.resolve()),
                   '--runtime-target', str(ROOT/'configs/targets/sm_90_h800.json')]
        with source.with_suffix('.log').open('w') as log:
            done = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
                TILEMEGA_F3_REPEAT=setting, TILEMEGA_F3_REPEAT_LAYERS=args.layers,
                TILEMEGA_F3_REPEAT_ARITHMETIC=arithmetic,
                TILEMEGA_F3_REGISTER_BUDGET=args.register_budget),
                stdout=log, stderr=subprocess.STDOUT)
        if arithmetic == '2':
            assert done.returncode and 'TILEMEGA_F3_REPEAT_ARITHMETIC must be0 or1' in source.with_suffix('.log').read_text()
            continue
        assert done.returncode == 0
        text = source.read_text()
        manifest = json.loads(Path(str(source)+'.plan.json').read_text())
        assert len(dispatch(manifest, build_defines(text, []), text)) == 63
        sources.append(text)
    invalid = args.out/'invalid-register.cu'
    command[3] = str(invalid.resolve())
    with invalid.with_suffix('.log').open('w') as log:
        done = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
            TILEMEGA_F3_REGISTER_BUDGET='4', TILEMEGA_F3_REPEAT_ARITHMETIC='0'),
            stdout=log, stderr=subprocess.STDOUT)
    assert done.returncode and 'TILEMEGA_F3_REGISTER_BUDGET must be0,1,2 or3' in invalid.with_suffix('.log').read_text()
    repeated = ',Stage_0,Repeat<' in sources[1]
    assert sources[2] == '#define TILEMEGA_F3_REPEAT_ARITHMETIC 1\n'+sources[1]
    assert repeated == (args.expect == 'repeat')
    if not repeated:
        assert 'F3 Repeat declined:' in sources[1]
        assert re.sub(r'// F3 Repeat declined:[^\n]*\n', '', sources[1]) == sources[0]
    print(json.dumps(dict(status='passed', evidence='verified', batch=args.batch,
                          layers=args.layers, repeated=repeated, logical_stages=63,
                          arithmetic_profile=True, invalid_arithmetic_rejected=True,
                          register_budget=args.register_budget, invalid_register_rejected=True)))


if __name__ == '__main__':
    main()
