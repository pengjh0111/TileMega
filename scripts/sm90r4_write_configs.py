#!/usr/bin/env python3
"""Write SM90R4 final serving configs (prompt §1.6, §10): one file per accepted cell.

Each records the solve command, the solver's selected tmexec coordinates, the
source HEAD and the generated-source and binary hashes, so the cell rebuilds
from one HEAD; `measured` holds the P-FINAL result and `correctness` the
P-CORR outcome. The schema follows configs/serving/sm90r3.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    p.add_argument('--solve-root', type=Path, required=True, help='folder holding <model>-B<b>/ solve outputs')
    p.add_argument('--accept-root', type=Path, required=True,
                   help='folder holding u-final-<cell>/ and u-corr-<cell>/ dispatcher units')
    p.add_argument('--baseline-source', default='new (R16 protocol, SM90R4)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for batch in a.batches:
        cell = f'{a.model}-B{batch}'
        folder = a.solve_root / cell
        bind = json.loads((folder / 'binding.json').read_text())
        receipt = json.loads((folder / 'decode.so.search/receipt.json').read_text())
        identity = json.loads((folder / 'decode.so.identity.json').read_text())
        solve = json.loads((folder / 'solve-command.json').read_text())
        if sha(folder / 'decode.so') != identity['binary_sha256']:
            raise ValueError(f'{cell}: binary differs from its identity record')
        final = json.loads((a.accept_root / f'u-final-{cell}/result.json').read_text())
        corr = json.loads((a.accept_root / f'u-corr-{cell}/result.json').read_text())
        unit = json.loads((a.accept_root / f'u-final-{cell}/unit.json').read_text())
        arm = next(iter(final['arms'].values()))
        if next(iter(final['spec']['arms'].values()))['decode_sha256'] != identity['binary_sha256']:
            raise ValueError(f'{cell}: P-FINAL measured a different binary')
        config = dict(
            schema=1, model=a.model, batch=batch, phase='decode', target='sm_90_h800',
            source_head=bind['head'], solve_command=solve['command'],
            compiler_sha256=solve['compiler_sha256'],
            coordinates=json.loads((folder / 'decode.so.tmexec.json').read_text()),
            winner=receipt['winner'], predicted_ns=bind['predicted_ns'],
            generated_source_sha256=identity['generated_source_sha256'],
            binary_sha256=identity['binary_sha256'], artifact_id=identity['artifact_id'],
            prefill=str(bind['prefill']), prefill_sha256=bind['prefill_sha256'],
            measured=dict(tpot_ms=arm['tpot_ms'], ttft_ms=arm['ttft_ms'], e2e_ms=arm['e2e_ms'],
                          tokens_per_s=arm['tokens_per_s'], process_range_pct=arm['range_pct'],
                          fresh_processes=arm['processes'], protocol='P-FINAL', gpu=unit['gpu'],
                          concurrency=final['spec'].get('concurrency'),
                          baseline_source=a.baseline_source),
            correctness=dict(c1=corr['c1_pass'], c2=corr['c2_pass'], pass_=corr['pass'],
                             hf64_divergence=corr.get('hf64_divergence'),
                             processes_token_equal=corr['processes_token_equal']))
        path = a.out / f'{a.model}_b{batch}.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        print(path.name)


if __name__ == '__main__':
    main()
