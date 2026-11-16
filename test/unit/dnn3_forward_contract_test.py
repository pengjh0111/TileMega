#!/usr/bin/env python3
"""Reject corrupted F3 manifests/source using a retained generated artifact."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.feature_contract import build_defines
from tilemega.dnn import f3_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--proposed-contract', type=Path)
    args = parser.parse_args()
    source = Path(str(args.artifact) + '.cu').read_text()
    manifest = json.loads(Path(str(args.artifact) + '.plan.json').read_text())
    defines = build_defines(source, [])
    contract = f3_contract
    if args.proposed_contract:
        spec = importlib.util.spec_from_file_location('proposed_forward_contract', args.proposed_contract)
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
    actual = contract.dispatch(manifest, defines, source)
    assert len(actual) == len(manifest['stages'])
    assert contract.implementations(manifest)['program'] == 'sm90_forward_f3'
    rejected = 0

    def reject(m=manifest, d=defines, s=source):
        nonlocal rejected
        try:
            contract.dispatch(m, d, s)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('corrupt F3 artifact was accepted')

    for key, value in [('phase', 'decode'), ('mode', 'L2'), ('loop', 0),
                       ('residency', 2), ('runtime_batch', 0), ('native_origin', 'unknown')]:
        bad = copy.deepcopy(manifest); bad[key] = value; reject(m=bad)
    for key, value in [('threads', 160), ('callee_limit', 255), ('requested_residency', 2),
                       ('runtime_occupancy_required', False)]:
        bad = copy.deepcopy(manifest); bad['register_contract'][key] = value; reject(m=bad)
    for key, value in [('schema', 2), ('abi', 'legacy'), ('residency', 2)]:
        bad = copy.deepcopy(manifest); bad['native_forward'][key] = value; reject(m=bad)
    for key, value in [('TILEMEGA_ARCH_ID', '800'), ('TILEMEGA_STATIC_FORWARD_F3', '0'),
                       ('TILEMEGA_STATIC_FORWARD_NATIVE', '1')]:
        reject(d=dict(defines, **{key: value}))
    corruptions = [
        source.replace('kF3ForwardIdentity[]', 'kOtherIdentity[]', 1),
        source + '\nusing HiddenLegacy=Sm90ForwardGemm<int>;\n',
        source.replace('Placement<Wrap_0,kOffset_0>', 'Placement<Wrap_1,kOffset_0>', 1),
        source.replace('SmallStage<Body_0,Bind_0,Shape_0>', 'GemmStage<Body_0,Bind_0,Shape_0>', 1),
        source.replace(',Stage_0,Stage_1', ',Stage_1,Stage_0', 1),
        source.replace('kF3ForwardMaps=', 'kWrongMaps=', 1),
        source + '\nusing Body_0=Sm90F3Layout<3,1>;\n',
    ]
    gemm = re.search(r'using Body_\d+=Sm90F3Gemm<(\d+),(\d+),(\d+),', source)
    assert gemm is not None
    corruptions.append(source[:gemm.start(3)] + '1' + source[gemm.end(3):])
    for damaged in corruptions:
        assert damaged != source
        reject(s=damaged)
    gather = next((i for i, s in enumerate(manifest['native_forward']['stages'])
                   if s['mode'] == 'gather'), None)
    if gather is not None:
        # Phase C may change the seed tile, but gather still requires convolution.
        for key, value in [('body', 'gemm'), ('tile_m', 32), ('tile_n', 16),
                           ('slots', 1), ('slots', 9), ('split_k', 0), ('split_k', 17)]:
            bad = copy.deepcopy(manifest)
            stage = bad['native_forward']['stages'][gather]
            stage[key] = value
            bad['stages'][gather]['coordinate'] = stage
            bad['stages'][gather]['body'] = stage['body']
            try:
                contract.validate(bad, defines)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError('invalid gather domain was accepted')
    print(f'F3 contract: {len(actual)} typed stages accepted; {rejected} corruptions rejected')


if __name__ == '__main__':
    main()
