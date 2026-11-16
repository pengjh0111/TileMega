"""Validate generated global-route calls, including per-layer operand aliases."""
import json
from pathlib import Path
import re
import sys
from tilemega.build.feature_contract import build_defines, validate_gemm_dispatch


def check(source):
    text = source.read_text()
    plan = Path(str(source) + '.plan.json')
    if not plan.exists():
        plan = source.with_suffix('.plan.json')
    manifest = json.loads(plan.read_text())
    def validate(value):
        return validate_gemm_dispatch(manifest, build_defines(value, []), value)
    def refused(value):
        assert value != text
        try:
            validate(value)
        except ValueError:
            return
        raise AssertionError('corrupt routing contract accepted')
    assert len(validate(text)) == len(manifest['gemms'])
    mode = manifest['native_dense']['route']
    fused = re.search(r'MoeFused<([^;]+)>;', text)
    args = fused[1].split(',')
    assert args[-1] == 'false'
    refused(text[:fused.start(1)] + ','.join(args[:-1]) + text[fused.end(1):])
    if mode == 'la':
        call = re.search(r'Moe<([^;]+)>;', text)
        args = call[1].split(','); assert args[-1] == 'true'
        refused(text[:call.start(1)] + ','.join(args[:-1]) + text[call.end(1):])
    elif mode == 'stage':
        call = re.search(r'MoeRoute<([^;]+)>;', text)
        args = call[1].split(',')
        router = re.search(r'Moe<([^;]+)>;', text)[1].split(',')
        # Independently emitted layer columns may alias the same route index.
        alias = [*args[:8], router[19], args[9]]
        validate(text[:call.start(1)] + ','.join(alias) + text[call.end(1):])
        corrupt = [*args[:8], router[14], args[9]]
        refused(text[:call.start(1)] + ','.join(corrupt) + text[call.end(1):])
        corrupt = [*args[:5], args[6], *args[6:]]
        refused(text[:call.start(1)] + ','.join(corrupt) + text[call.end(1):])
    else:
        raise AssertionError('expected a global routing variant')
    print(json.dumps(dict(source=str(source), mode=mode, passed=True)))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        check(Path(name))
