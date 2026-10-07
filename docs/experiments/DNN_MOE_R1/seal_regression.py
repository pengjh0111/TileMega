#!/usr/bin/env python3
"""Seal the actual baseline's eight rebuilt plans, independently of G-REG."""
import argparse
import hashlib
import json
from pathlib import Path

from identity_dm import sha, verify
from regression_compare import sass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for model in ('llama', 'qwen3'):
        for batch in (1, 16):
            cell = f'{model}_B{batch}'
            base = args.reference / cell
            tokens = json.loads((base / 'tokens.json').read_text())
            if (tokens['steps'] != 64 or len(tokens['tokens']) != batch or
                    any(len(row) != 64 for row in tokens['tokens'])):
                raise ValueError('reference token smoke is incomplete: ' + cell)
            for phase in ('prefill', 'decode'):
                so = base / phase / 'plan.so'
                identity = verify(so)
                if identity['source']['head'] != args.head:
                    raise ValueError('reference HEAD drifted')
                if identity['source']['diff_sha256'] != hashlib.sha256(b'').hexdigest():
                    raise ValueError('reference worktree was modified')
                rows.append(dict(cell=cell, phase=phase, binary=str(so),
                                 artifact_id=identity['artifact_id'],
                                 cu_sha256=identity['cu_sha256'],
                                 so_sha256=identity['so_sha256'],
                                 sass_sha256=hashlib.sha256(sass(so).encode()).hexdigest(),
                                 resources=identity['kernels'],
                                 tokens_sha256=sha(base / 'tokens.json')))
    bank = dict(evidence='verified', baseline_head=args.head, plans=rows,
                scope='reference artifacts and 64-step smoke; ctest remains a separate gate')
    bank['seal_sha256'] = hashlib.sha256(json.dumps(
        bank, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError('refuse to overwrite a sealed bank')
    args.out.write_text(json.dumps(bank, indent=2) + '\n')
    print(json.dumps(dict(event='reference_bank_sealed', plans=8,
                          seal_sha256=bank['seal_sha256'])), flush=True)


if __name__ == '__main__':
    main()
