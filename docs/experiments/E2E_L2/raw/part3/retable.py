#!/usr/bin/env python3
"""Control variants for the Part 3.1 attribution.

Rewrites only the `kDependencies`/`kDependencyOffsets` tables of a generated
.cu, leaving every other line -- and the whole harness -- byte identical, so
the table is the single variable.

  kall     force every map to kAll, same edge set (isolates the map's cost)
  reduced  force kAll *and* re-run the transitive reduction with every edge
           eligible as an intermediate, which is what the generator did before
           narrowed edges had to be excluded from it
"""
import re, sys

src, mode, dst = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src).read()
block = re.search(r'constexpr StageDependency kDependencies\[\] = \{\n(.*?)\n\};',
                  text, re.S)
rows = re.findall(r'\{(\d+)u, (\d+)u, StageDependency::Map::k(\w+), (\d+)u, '
                  r'(-?\d+), (-?\d+), (\d+)u\}', block.group(1))
edges = [(int(p), int(c)) for p, c, *_ in rows]
stage_count = len(re.search(r'kDependencyOffsets\[\] = \{\n(.*?)\n\};',
                            text, re.S).group(1).split(',')) - 2

if mode == 'reduced':
    incoming = {c: [] for _, c in edges}
    for p, c in edges: incoming[c].append(p)
    reaches = [[False] * stage_count for _ in range(stage_count)]
    for c in range(stage_count):
        for p in incoming.get(c, []):
            reaches[c][p] = True
            for s in range(stage_count):
                if reaches[p][s]: reaches[c][s] = True
    edges = [(p, c) for p, c in edges
             if not any(o != p and reaches[o][p] for o in incoming[c])]

edges.sort(key=lambda e: (e[1], e[0]))
body = '\n'.join(f'  {{{p}u, {c}u, StageDependency::Map::kAll, 1u, 0, 0, 1u}},'
                 for p, c in edges)
offsets, at = [], 0
for stage in range(stage_count + 1):
    offsets.append(at)
    while at < len(edges) and edges[at][1] == stage: at += 1
text = text[:block.start()] + \
    'constexpr StageDependency kDependencies[] = {\n' + body + '\n};' + \
    text[block.end():]
text = re.sub(r'constexpr std::uint32_t kDependencyOffsets\[\] = \{\n.*?\n\};',
              'constexpr std::uint32_t kDependencyOffsets[] = {\n  ' +
              ' '.join(f'{o}u,' for o in offsets) + '\n};', text, flags=re.S)
text = re.sub(r'kDependencies, \d+u', f'kDependencies, {len(edges)}u', text)
open(dst, 'w').write(text)
print(f'{mode}: {len(rows)} -> {len(edges)} edges  {dst}')
