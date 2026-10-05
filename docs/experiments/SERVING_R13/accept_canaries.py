#!/usr/bin/env python3
"""Accept the two preregistered whole-round replacements, without erasing raw data."""
import argparse, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = ('llama_B1', 'qwen3_B16')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def accept(out):
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    # Validate both replacements before invalidating either original round.
    for cell in CELLS:
        old_dir = HERE/'raw'/f'D2_{cell}_r2'
        new_dir = HERE/'raw'/f'D2_canary_{cell}_r2'
        old = old_dir/cell/'round2.json'
        new = new_dir/cell/'round2.json'
        assert read(new_dir/'guard_result.json')['code'] == 0, 'replacement guard failed'
        original, replacement = read(old), read(new)
        assert not replacement.get('invalidated'), 'replacement invalidated'
        assert original['order'] == replacement['order'], 'order changed'
        assert set(replacement['arms']) == set(original['arms']), 'arm set changed'
        for record in replacement['arms'].values():
            assert record['exit_code'] == 0 and record['e2e_seconds'] > 0, 'replacement arm failed'
        old_arms, new_arms = read(old_dir/'arms.json')[cell], read(new_dir/'arms.json')[cell]
        assert old_arms == new_arms, 'binary or execution definitions changed'
        for arm in new_arms:
            for phase, binary in arm.get('binaries', {}).items():
                if binary.get('sha256'):
                    assert sha(Path(arm[phase])) == binary['sha256'], 'binary changed'
        for label in ('B0-D', 'R13F'):
            tokens = Path(replacement['arms'][label]['out'])/'tokens_N1024_run1.json'
            reference = Path(original['arms'][label]['out'])/'tokens_N1024_run1.json'
            assert read(tokens) == read(reference), 'replacement tokens changed'
        backup = out/f'{cell}_original_round2.json'
        if not backup.exists():
            backup.write_bytes(old.read_bytes())
        checks.append(dict(cell=cell, original=str(old), original_sha256=sha(backup),
                           replacement=str(new), replacement_sha256=sha(new),
                           tokens_equal=True, binary_definitions_equal=True))
    for check in checks:
        old = Path(check['original'])
        data = read(old)
        data.update(invalidated=True, superseded_by=check['replacement'],
                    invalidation_reason='one preregistered >2% canary rerun')
        old.write_text(json.dumps(data, indent=2)+'\n')
    (out/'acceptance.json').write_text(json.dumps(dict(pass_=True, rounds=checks), indent=2)+'\n')
    print('Accepted two clean same-binary canary replacement rounds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    accept(parser.parse_args().out)
