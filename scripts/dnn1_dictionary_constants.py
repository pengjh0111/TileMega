#!/usr/bin/env python3
"""Share schema2 selector field names without changing measured values or IDs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def encode(section):
    if section.get('schema') != 2:
        raise ValueError('schema2 input required')
    result = copy.deepcopy(section)
    layouts, indices, keys = [], {}, []
    for metric, key, unit in section['keys']:
        pairs = [field.split('=', 1) for field in key.split(';')]
        if any(len(pair) != 2 or not all(pair) for pair in pairs):
            raise ValueError('invalid selector')
        names = tuple(name for name, value in pairs)
        if len(set(names)) != len(names):
            raise ValueError('duplicate selector field')
        layout = (metric, unit, names)
        if layout not in indices:
            indices[layout] = len(layouts)
            layouts.append([metric, unit, list(names)])
        keys.append([indices[layout], *(value for name, value in pairs)])
    result.update(schema=3, key_layouts=layouts, keys=keys)
    restored = [[layouts[row[0]][0], ';'.join(name+'='+value for name, value in
        zip(layouts[row[0]][2], row[1:])), layouts[row[0]][1]] for row in keys]
    if restored != [list(row) for row in section['keys']]:
        raise ValueError('selector round trip changed the source')
    return result


def convert(source, output):
    original = source.read_text()
    marker = ',\n  "sm90_forward_constants": '
    if original.count(marker) != 1:
        raise ValueError('expected one forward section in the original target format')
    target = json.loads(original)
    result = encode(target['sm90_forward_constants'])
    prefix = original.split(marker, 1)[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(prefix+marker+json.dumps(result, separators=(',', ':'))+'\n}\n')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(source_sha256=sha(source), output_sha256=sha(output),
        keys=len(result['keys']), layouts=len(result['key_layouts']),
        medians=sum(map(len, result['median_by_gpu'].values())),
        calibration_variant_unchanged=True, nonforward_prefix_unchanged=True,
        csv_scope='original constant_id CSV is unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = convert(args.source, args.output)
    args.output.with_suffix('.conversion.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
