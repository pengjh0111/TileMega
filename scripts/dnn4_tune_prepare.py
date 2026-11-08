#!/usr/bin/env python3
"""Prepare the six DT-MODEL in-situ variants from legal solver candidates."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def arguments(value):
    parts, start, depth = [], 0, 0
    for i, char in enumerate(value):
        if char == '<':
            depth += 1
        elif char == '>':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(value[start:i])
            start = i + 1
    parts.append(value[start:])
    return parts


def signatures(source, coordinates):
    bodies = {int(i): (kind, arguments(args)) for i, kind, args in
              re.findall(r'^using Body_(\d+)=(\w+)<(.*)>;$', source, re.M)}
    structs = dict(re.findall(r'struct ((?:Conv|Pool)_\d+)\s*\{(.*?)\n\};', source, re.S))
    types = [bodies[i][0] for i in range(len(coordinates['stages']))]
    result = []
    for i, coordinate in enumerate(coordinates['stages']):
        kind, args = bodies[i]
        if kind == 'Sm90F3Gemm':
            # NStore describes the logical width; padded N depends on the tile.
            shape, chain, extras = (args[5], args[8], args[7]), args[9], args[10:12]
        elif kind == 'Sm90F3GemmSwapAB':
            shape, chain, extras = tuple(args[2:6]), args[6], []
        else:
            result.append((kind, i))
            continue
        chain = re.sub(r'Sm90F3SideLayout<\d+,\d+,', 'Sm90F3SideLayout<TM,TN,', chain)
        geometry = tuple(re.sub(r'\s+', '', structs.get(name, name)) for name in extras)
        neighbors = (types[i-1] if i else 'entry', types[i+1] if i+1 < len(types) else 'exit')
        result.append((kind, coordinate['mode'], shape, chain, geometry, neighbors))
    return result


def prepare(receipt, source):
    rank0 = receipt['ranked'][0]
    coordinates = rank0['coordinates']
    prediction = rank0['prediction']
    stages = prediction['search_receipt']['stages']
    groups = {}
    for i, signature in enumerate(signatures(source, coordinates)):
        if coordinates['stages'][i]['body'] not in ('gemm', 'conv'):
            continue
        groups.setdefault(signature, []).append(i)
    targets = []
    for members in groups.values():
        weight = sum(prediction['stages'][i]['stage_ns'] for i in members)
        if weight < .02 * prediction['forward_ns']:
            continue
        # A coordinate must occur in every member's enumerated legal pool.
        pools = [stages[i]['tuning_candidates'] for i in members]
        options = []
        for candidate in pools[0]:
            coordinate = candidate['coordinate']
            matches = [next((c for c in pool if c['coordinate'] == coordinate), None) for pool in pools]
            if all(matches):
                options.append(dict(coordinate=coordinate, prediction_ns=sum(c['prediction_ns'] for c in matches)))
        if not options:
            raise ValueError('no common legal coordinate for grouped stages ' + str(members))
        targets.append(dict(stages=members, prediction_ns=weight, share=weight/prediction['forward_ns'], candidates=options))
    targets.sort(key=lambda g: (-g['prediction_ns'], g['stages'][0]))
    targets = targets[:16]
    variants = []
    for rank in range(6):
        program = copy.deepcopy(coordinates)
        for group in targets:
            candidate = group['candidates'][min(rank, len(group['candidates'])-1)]['coordinate']
            for stage in group['stages']:
                program['stages'][stage] = candidate
        variants.append(program)
    for index, group in enumerate(targets):
        group.update(group=index, distinct_candidates=len(group['candidates']),
                     repeated_tail=len(group['candidates']) < 6)
    return dict(protocol='P-TUNE;DT-MODEL six choices', rank0=coordinates, groups=targets,
                variants=variants, predicted_ns=prediction['forward_ns'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = prepare(json.loads(args.receipt.read_text()), args.source.read_text())
    result['inputs'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (args.receipt, args.source, Path(__file__))}
    for rank, coordinates in enumerate(result.pop('variants')):
        save(args.out / f'variant-{rank}.json', coordinates)
    save(args.out / 'result.json', result)
    print(json.dumps(dict(groups=len(result['groups']), stages=sum(len(g['stages']) for g in result['groups']),
                          candidate_counts=[g['distinct_candidates'] for g in result['groups']])))


if __name__ == '__main__':
    main()
