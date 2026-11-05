#!/usr/bin/env python3
"""Route every frozen DNN1 calibration key to a portable probe recipe."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

RECIPES = {'pool_finish': 'pool', 'rowstat_finish': 'rowstat',
           'sca_panel_64_128': 'sca', 'sca_panel_32': 'sca'}


def catalog(target):
    section = json.loads(target.read_text())['sm90_forward_constants']
    schema = section['schema']
    if schema == 2:
        keys = section['keys']
    elif schema == 3:
        keys = []
        for row in section['keys']:
            metric, unit, fields = section['key_layouts'][row[0]]
            if len(row) != len(fields)+1:
                raise ValueError('dictionary key field count mismatch')
            keys.append([metric, ';'.join(n+'='+v for n, v in zip(fields, row[1:])), unit])
    else:
        raise ValueError('a compact measured epoch is required')
    sources = section['calibration_epoch']['source_recipes']
    by_sha = {sha: name for name, sha in sources.items()}
    if len(by_sha) != len(sources):
        raise ValueError('ambiguous recipe provenance')
    origins = ['base']*len(keys)
    assigned = set()
    for supplement in section['supplements']:
        name = by_sha[supplement['raw_sources']['sm90_forward_constants.json']['sha256']]
        for index in supplement['key_indices']:
            if type(index) is not int or not 0 <= index < len(keys) or index in assigned:
                raise ValueError('duplicate or invalid recipe source index')
            origins[index] = name
            assigned.add(index)
    groups = collections.defaultdict(list)
    for index, ((metric, key, unit), name) in enumerate(zip(keys, origins)):
        groups[name].append(dict(constant_id=index, metric=metric, key=key, unit=unit))
    return dict(status='CPU method inventory; not new calibration',
        target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        keys=len(keys), medians=sum(map(len, section['median_by_gpu'].values())),
        recipes={name: dict(generator='dnn1_forward_primitive_probe.py' if name in RECIPES
            else 'dnn1_forward_recipe_probe.py', family=RECIPES.get(name, name), keys=rows)
            for name, rows in groups.items()},
        limitations=['Production header identity belongs to each build receipt; generating the same CUDA source does not recreate historical header snapshots.',
            'Changed production implementations require new guarded measurements.',
            'Composite register Apply chains require their own measured keys; this inventory only covers the frozen epoch.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = catalog(args.target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
