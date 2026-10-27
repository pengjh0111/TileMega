#!/usr/bin/env python3
"""Write the SM90R3 final serving configs (prompt §1.6 item 4, §9 item 2).

One file per measured cell under configs/serving/sm90r3/, recording the solve
command, the solver's selected tmexec coordinates, the source HEAD and the
hashes of the generated source and binary, so a single HEAD reproduces it.
"""
import argparse
import hashlib
import json
from pathlib import Path

R = Path('/root/TileMega/runs/sm90r3')
M = Path('/root/TileMega-m')
DENSE = ('qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_1b', 'llama3_2_3b', 'llama3_1_8b')
MOE = 'qwen3_30b_a3b'
BATCHES = (1, 2, 4, 8, 16)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=M / 'configs/serving/sm90r3')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    written, missing = [], []
    for model in (*DENSE, MOE):
        for batch in BATCHES:
            cell = f'{model}-B{batch}'
            folder = M / f'runs/sm90r3/solve/FINAL/{cell}'
            binding = folder / 'binding.json'
            if not binding.exists():
                missing.append(cell)
                continue
            bind = json.loads(binding.read_text())
            receipt = json.loads((folder / 'decode.so.search/receipt.json').read_text())
            identity = json.loads((folder / 'decode.so.identity.json').read_text())
            solve = json.loads((folder / 'solve-command.json').read_text())
            coordinates = json.loads((folder / 'decode.so.tmexec.json').read_text())
            result = R / f'v/results/final-v-{cell}.json'
            corr = R / f'v/results/corr-v-{cell}.json'
            config = dict(
                schema=1, model=model, batch=batch, phase='decode', target='sm_90_h800',
                source_head=bind['head'],
                solve_command=solve['command'], compiler_sha256=solve['compiler_sha256'],
                coordinates=coordinates,
                winner=receipt['winner'],
                predicted_ns=bind['predicted_ns'],
                generated_source_sha256=identity['generated_source_sha256'],
                binary_sha256=identity['binary_sha256'],
                artifact_id=identity['artifact_id'],
                prefill=str(bind['prefill']), prefill_sha256=bind['prefill_sha256'],
                prefill_kind='rebuilt R2 closeout options at the final HEAD' if model != MOE
                else 'frozen R1 legacy MoE prefill (gamma-folded expert pages)',
                stages=int(coordinates.get('stage_count', 0)) or None)
            if result.exists():
                measured = json.loads(result.read_text())
                arm = next(iter(measured['arms'].values()))
                config['measured'] = dict(tpot_ms=arm['tpot_ms'], ttft_ms=arm['ttft_ms'],
                                          e2e_ms=arm['e2e_ms'], tokens_per_s=arm['tokens_per_s'],
                                          process_range_pct=arm['range_pct'],
                                          fresh_processes=arm['processes'], protocol='P-FINAL')
            if corr.exists():
                check = json.loads(corr.read_text())
                config['correctness'] = dict(c1=check['c1_pass'], c2=check['c2_pass'],
                                             hf64_recorded=check.get('hf64_pass'),
                                             processes_token_equal=check['processes_token_equal'])
            path = a.out / f'{model}_b{batch}.json'
            path.write_text(json.dumps(config, indent=2) + '\n')
            written.append(path.name)
    (a.out / 'README.md').write_text(
        '# SM90R3 final serving configurations\n\n'
        'One file per accepted cell (25 dense + 5 MoE). Each records the solve command, the\n'
        "solver's selected tmexec coordinates, the source HEAD, and the generated-source and\n"
        'binary hashes, so the cell rebuilds from a single HEAD. `measured` holds the P-FINAL\n'
        'result and `correctness` the P-CORR outcome. Dense prefills are the R2 closeout\n'
        'options rebuilt at this HEAD; the MoE prefill stays the frozen R1 legacy binary whose\n'
        'expert pages match the native decode.\n')
    print(f'wrote {len(written)} configs' + (f'; missing {missing}' if missing else ''))


if __name__ == '__main__':
    main()
