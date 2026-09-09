#!/usr/bin/env python3
"""Check the shared runtime-plan refactor against a preserved pre-change compiler."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    current = repo/'build-portable/tools/tilemega-compile'
    output = args.out or here/'runtime_projection'
    output.mkdir(parents=True,exist_ok=True)
    if args.out and (output/'codegen_equivalence.json').exists():
        raise RuntimeError('refusing to overwrite codegen evidence')
    rows = []
    with tempfile.TemporaryDirectory(prefix='tilemega-projection-codegen-') as tmp:
        for model in ('gqa2','mha4'):
            for variants in (False, True):
                texts = []
                for label, compiler in (('before', args.control), ('after', current)):
                    destination = Path(tmp)/f'{model}_{variants}_{label}.cu'
                    command = [str(compiler), str(repo/f'docs/experiments/SEQSCAN/raw/export/{model}.json'),
                               str(destination)]
                    if variants:
                        command += ['--variants', str(repo/'docs/experiments/OWNERSHIP/plan_structured.json')]
                    result = subprocess.run(command, capture_output=True, text=True)
                    (output/f'{model}_{variants}_{label}.log').write_text(result.stdout+result.stderr)
                    result.check_returncode()
                    texts.append(destination.read_bytes())
                rows.append(dict(model=model, variants=variants, equal=texts[0] == texts[1],
                                 sha256=[hashlib.sha256(text).hexdigest() for text in texts],
                                 bytes=[len(text) for text in texts]))
                (output/'codegen_equivalence.json').write_text(json.dumps(rows, indent=2)+'\n')
                if texts[0] != texts[1]:
                    raise RuntimeError(f'generated code changed: {model} variants={variants}')


if __name__ == '__main__':
    main()
