#!/usr/bin/env python3
"""Matched fixed-geometry request check for the single-page loader edit."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from measure_pg_ablation import LOCK, cell_sources, manifest, power_policy, sha
from measure_page_vector_endpoints import CELLS, FIELDS, run_measure

WORK = Path('/root/r11_work/pagecheck_once')


def main() -> None:
    cases = json.loads((WORK / 'cases.json').read_text())
    checks = [json.loads((WORK / name / 'summary.json').read_text())
              for name in ('fresh50_llama_B1', 'fresh50_remaining')]
    if len(cases) != len(CELLS) or not all(c.get('complete') and c.get('failed') == 0
                                           for c in checks) or sum(c['passed'] for c in checks) != 200:
        raise RuntimeError('four fresh-process checks must pass before timing')
    out = WORK / 'matched_e2e'
    out.mkdir(exist_ok=True)
    policy = power_policy(out)
    results = []
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, (model, batch) in enumerate(CELLS):
            case = cases[index]
            if case['batch'] != batch or model not in case['model']:
                raise RuntimeError('case ordering changed')
            source, prefill = cell_sources(model, batch)
            off, pages = source['off'], Path(case['decode'])
            if sha(pages) != case['binary_sha256']['decode']:
                raise RuntimeError(f'protocol binary differs from timed binary: {model} B{batch}')
            a, b = manifest(off), manifest(pages)
            if any(a[field] != b[field] for field in FIELDS) or a['pg'] != 'off' or b['pg'] != 'pages':
                raise RuntimeError(f'PG geometry differs: {model} B{batch}')
            order = (('off', off), ('pages', pages)) if index % 2 == 0 else (
                ('pages', pages), ('off', off))
            for label, binary in order:
                row = run_measure(model, batch, label, binary, prefill, policy,
                                  out / f'{model}_B{batch}' / label)
                results.append(row)
                (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
