#!/usr/bin/env python3
"""Attribute the instrumented PG-1 realized chain to emitted stage kinds."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import re
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'docs/experiments/SERVING_R11/page_diagnostics/raw.tar.xz'


def stage_kinds(cu: Path) -> list[str]:
    source = cu.read_text()
    stage_lines = source.split('constexpr StageDesc kStages[] = {', 1)[1].split('};', 1)[0]
    gemm_ids = [int(g) for g in re.findall(r'TaskKind::kGemm,\s*(\d+)u', stage_lines)]
    head_id = max(gemm_ids)
    result = []
    for line in stage_lines.splitlines():
        match = re.search(r'TaskKind::(\w+),\s*(\d+)u', line)
        if not match:
            continue
        kind, identifier = match.group(1), int(match.group(2))
        if kind == 'kGemm':
            result.append('lm_head' if identifier == head_id else
                          ('qkv', 'o', 'gate_up', 'down')[identifier % 4])
        elif kind == 'kGemmCombine':
            result.append('combine')
        else:
            result.append(kind.removeprefix('k').lower())
    return result


def main() -> None:
    output = ROOT / 'docs/experiments/SERVING_R11/page_diagnostics/pg_chain_classes.tsv'
    records = []
    with tarfile.open(ARCHIVE) as source:
        for model in ('llama', 'qwen3'):
            for batch in (1, 16):
                cell = f'{model}_B{batch}'
                cu = Path(f'/root/r11_work/protocol/page_v2/{cell}_pages/plan.so.cu')
                kinds = stage_kinds(cu)
                chain = source.extractfile(f'results/{cell}/chain/trace_v2.chain_links.tsv')
                if chain is None:
                    raise FileNotFoundError(f'chain trace for {cell}')
                groups: dict[str, list[int]] = {}
                for item in csv.DictReader(io.TextIOWrapper(chain), delimiter='\t'):
                    stage = int(item['stage'])
                    groups.setdefault(kinds[stage], []).append(int(item['wall_ns']))
                for kind, wall in sorted(groups.items()):
                    records.append(dict(model=model, batch=batch, stage_kind=kind,
                                        links=len(wall), wall_total_ns=sum(wall),
                                        wall_median_ns=statistics.median(wall)))
    with output.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(), delimiter='\t')
        writer.writeheader()
        writer.writerows(records)
    print(output)


if __name__ == '__main__':
    main()
