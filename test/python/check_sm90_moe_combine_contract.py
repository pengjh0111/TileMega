"""Check the standalone combine binds its expert partials and residual."""
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
        return validate_gemm_dispatch(manifest,build_defines(value,[]),value)
    assert len(validate(text)) == len(manifest['gemms'])
    call = re.search(r'MoeCombine<([^;]+)>;',text)
    args = call[1].split(',')
    corruptions = [[*args[:4],args[5],args[4]], [args[0],args[1],str(int(args[2])*2),*args[3:]]]
    for bad in corruptions:
        mutated = text[:call.start(1)] + ','.join(bad) + text[call.end(1):]
        try:
            validate(mutated)
        except ValueError:
            continue
        raise AssertionError('accepted corrupt combine binding')
    producer = list(re.finditer(r'(?:MoeFused|Moe)<([^;]+)>;',text[:call.start()]))[-1]
    args = producer[1].split(',');assert args[-1] == 'true'
    mutated = text[:producer.start(1)] + ','.join(args[:-1]) + text[producer.end(1):]
    try:
        validate(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted an enabled LA combine plus standalone combine')
    print(json.dumps(dict(source=str(source),passed=True)))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        check(Path(name))
