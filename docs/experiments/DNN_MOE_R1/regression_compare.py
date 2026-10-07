#!/usr/bin/env python3
"""Require exact CUDA, every ptxas resource context, SASS, and smoke tokens."""
import argparse
import json
from pathlib import Path
import subprocess

from identity_dm import verify


def sass(binary):
    text = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '-sass', str(binary)],
                                   text=True)
    # Only the enclosing binary filename is host-path dependent. Keep all
    # function names, addresses, instructions, and control words verbatim.
    return '\n'.join(line for line in text.splitlines()
                     if not line.startswith('Fatbin elf code:') and not line.startswith('file = '))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--partial', action='store_true',
                        help='collect available evidence; missing ctest never passes G-REG')
    args = parser.parse_args()
    rows = []
    for model in ('llama', 'qwen3'):
        for batch in (1, 16):
            cell = f'{model}_B{batch}'
            base = args.root / 'reference' / cell
            new = args.root / 'candidate' / cell
            tokens_old = json.loads((base / 'tokens.json').read_text())
            tokens_new = json.loads((new / 'tokens.json').read_text())
            for phase in ('prefill', 'decode'):
                old = base / phase / 'plan.so'
                candidate = new / phase / 'plan.so'
                oi, ni = verify(old), verify(candidate)
                rows.append(dict(cell=cell, phase=phase,
                                 reference_identity=oi['artifact_id'],
                                 candidate_identity=ni['artifact_id'],
                                 cu_equal=Path(str(old) + '.cu').read_bytes() ==
                                          Path(str(candidate) + '.cu').read_bytes(),
                                 resources_equal=oi['kernels'] == ni['kernels'],
                                 sass_equal=sass(old) == sass(candidate),
                                 token_equal=tokens_old == tokens_new))
    ctest = {}
    for side in ('reference', 'candidate'):
        path = args.root / side / 'ctest.json'
        if args.partial and not path.exists():
            ctest[side] = dict(exit_code=None, status='not run; native test build failed')
        else:
            ctest[side] = json.loads(path.read_text())
    artifacts_passed = all(all(r[k] for k in
                              ('cu_equal', 'resources_equal', 'sass_equal', 'token_equal'))
                           for r in rows)
    passed = artifacts_passed and all(r['exit_code'] == 0 for r in ctest.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(gate='G-REG', passed=passed,
                                        available_checks_passed=artifacts_passed, rows=rows,
                                        ctest=ctest, evidence='verified'), indent=2) + '\n')
    print('G-REG ' + ('PASS' if passed else 'FAIL'), flush=True)
    if not passed and not (args.partial and artifacts_passed):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
