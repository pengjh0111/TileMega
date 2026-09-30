#!/usr/bin/env python3
"""Attribute the instrumented PG-1 realized chain to emitted stage kinds."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'docs/experiments/SERVING_R11/page_diagnostics/raw.tar.xz'


def stage_kinds(cu: Path, manifest: Path | None = None) -> list[str]:
    source = cu.read_text()
    stage_lines = source.split('constexpr StageDesc kStages[] = {', 1)[1].split('};', 1)[0]
    gemm_ids = [int(g) for g in re.findall(r'TaskKind::kGemm,\s*(\d+)u', stage_lines)]
    head_id = max(gemm_ids)
    splits = {g['index']:g['split_k'] for g in json.loads(manifest.read_text())['gemms']} if manifest else {}
    result = []
    for line in stage_lines.splitlines():
        match = re.search(r'TaskKind::(\w+),\s*(\d+)u', line)
        if not match:
            continue
        kind, identifier = match.group(1), int(match.group(2))
        if kind == 'kGemm':
            result.append('lm_head' if identifier == head_id else
                          ('qkv', 'o', 'gate_up', 'down')[identifier % 4])
            if splits.get(identifier,1)>1:result.append('combine')
        elif kind == 'kGemmCombine':
            result.append('combine')
        else:
            result.append(kind.removeprefix('k').lower())
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--cu',type=Path);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--chain',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.cu:
        kinds=stage_kinds(args.cu,args.manifest)
        groups={}
        with args.chain.open() as stream:
            for row in csv.DictReader(stream,delimiter='\t'):
                stage=int(row['stage']);kind=kinds[stage]
                groups.setdefault(kind,[]).append(int(row['wall_ns']))
        rows=[dict(stage_kind=k,links=len(v),wall_total_ns=sum(v),wall_median_ns=statistics.median(v)) for k,v in groups.items()]
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['stage_kind','links','wall_total_ns','wall_median_ns'],delimiter='\t');writer.writeheader();writer.writerows(rows)
        return
    output = ROOT / 'docs/experiments/SERVING_R11/page_diagnostics/pg_chain_classes.tsv'
    records = []
    with tarfile.open(ARCHIVE) as source:
        for model in ('llama', 'qwen3'):
            filename = 'llama_config_public_copy.json' if model == 'llama' else 'qwen_config.json'
            config = json.loads((ROOT / 'docs/experiments/MODELS/sources' / filename).read_text())
            hidden = config['hidden_size']
            layers = config['num_hidden_layers']
            qkv_width = (config['num_attention_heads'] +
                         2 * config['num_key_value_heads']) * config['head_dim']
            weight_bytes = {
                'qkv': layers * qkv_width * hidden * 2,
                'o': layers * hidden * hidden * 2,
                'gate_up': layers * 2 * config['intermediate_size'] * hidden * 2,
                'down': layers * hidden * config['intermediate_size'] * 2,
                'lm_head': config['vocab_size'] * hidden * 2,
            }
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
                                        wall_median_ns=statistics.median(wall),
                                        weight_bytes=weight_bytes.get(kind, ''),
                                        effective_weight_gbps=(weight_bytes[kind] / sum(wall)
                                                               if kind in weight_bytes else '')))
    with output.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(), delimiter='\t')
        writer.writeheader()
        writer.writerows(records)
    print(output)


if __name__ == '__main__':
    main()
