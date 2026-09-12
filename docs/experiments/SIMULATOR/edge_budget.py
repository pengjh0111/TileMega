#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Why S1-c fails, and by how much it could be fixed.

Per-Plan evaluation is edge-dominated: two mandatory sweeps over the runtime
DAG (in-degree, then readiness propagation).  This counts the edges by the
dependency map that produced them, against the count a barrier node per
`kAll` group would need, so the S1-c degradation plan carries a number instead
of an intention.

The compression is NOT applied.  `MaterializeRuntimeTaskGraph` is shared with
the host, and its expanded edge count is what `E2E_PLACE_STATS` reports and
what E1-b pins byte for byte (H2); a simulator-local rewrite is possible but
is a structural change made while the simulator is frozen for evaluation (H8).
"""
import collections, re, sys


def main(cu_path, dump_path):
    body = re.search(r'constexpr StageDependency kDependencies0\[\] = \{(.*?)\n\};',
                     open(cu_path).read(), re.S).group(1)
    counts = {}
    for line in open(dump_path):
        if line.startswith('E2E_PLACE_BASE'):
            fields = dict(kv.split('=') for kv in line.split()[1:])
            counts[int(fields['stage'])] = int(fields['active'])

    expanded, compressed = collections.Counter(), collections.Counter()
    for line in body.strip().splitlines():
        tokens = re.findall(r'[\w:]+', line)
        if not tokens or not tokens[0].rstrip('u').isdigit():
            continue
        producer, consumer = (int(tokens[i].rstrip('u')) for i in (0, 1))
        kind = 'kAll' if 'kAll' in line else 'kWindow' if 'kWindow' in line else 'kIdentity'
        np_, nc = counts.get(producer, 0), counts.get(consumer, 0)
        if kind == 'kAll':
            expanded[kind] += np_ * nc
            compressed[kind] += np_ + nc  # every producer -> barrier -> every consumer
        else:
            fan = nc * max(1, min(int(tokens[-1].rstrip('u')), np_))
            expanded[kind] += fan
            compressed[kind] += fan

    print(f'{"map":12s} {"as expanded":>12s} {"with barriers":>14s}')
    for kind in ('kIdentity', 'kWindow', 'kAll'):
        print(f'{kind:12s} {expanded[kind]:12,} {compressed[kind]:14,}')
    total, small = sum(expanded.values()), sum(compressed.values())
    print(f'{"TOTAL":12s} {total:12,} {small:14,}')
    print(f'\nkAll is {100.0 * expanded["kAll"] / total:.1f}% of the edge budget; '
          f'compressing it is {total / small:.0f}x fewer edges to sweep.')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:3]))
