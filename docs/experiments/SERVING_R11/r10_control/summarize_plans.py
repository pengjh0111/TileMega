#!/usr/bin/env python3
"""Summarize the eight immutable-baseline R10 plan builds without rerunning them."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    output = []
    for model in ('llama', 'qwen3'):
        for phase in ('prefill', 'decode'):
            for batch in (1, 16):
                name = f'{model}_{phase}_B{batch}'
                directory = ROOT / 'plans' / name
                result = json.loads((directory / 'result.json').read_text())
                manifest = json.loads((directory / 'plan.so.plan.json').read_text())
                sass = json.loads((directory / 'fp64_audit.json').read_text())
                measurements = list(csv.DictReader(
                    (directory / 'plan.so.top3_measured.tsv').open(), delimiter='\t'))
                best = min(measurements, key=lambda row: float(row['mean_ms']))
                output.append(dict(cell=name, source=result['source_sha256'],
                    solve_seconds=result['seconds'], mode=manifest['mode'],
                    top3_fastest_ms=best['mean_ms'], top3_rank=best['rank'],
                    top3_mode=best['mode'], fp64_instructions=sass['fp64_total'],
                    passed=result['returncode'] == 0 and sass['fp64_total'] == 0))
    with (ROOT / 'plans_summary.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys(), delimiter='\t')
        writer.writeheader(); writer.writerows(output)
    (ROOT / 'plans_summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
