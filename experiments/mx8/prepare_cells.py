"""Resolve the nine immutable main-experiment references for MetaX AB4c."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import sha


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p))


def main():
    out = ROOT / 'runs/mx8/cells'
    out.mkdir(parents=True, exist_ok=True)
    for batch in (1, 4, 16):
        directory = ROOT / f'runs/mx5/consumed_prefetch_qwen3_1_7b_B{batch}_L1prefill_full_top4_execution_v1/execution'
        selection = json.loads((directory / 'selection.json').read_text())
        index = selection['selected_index']
        candidate = next(c for c in selection['candidates'] if c['index'] == index)
        command = json.loads((directory / f'candidate{index}/round0/process.json').read_text())['argv']
        decode, prefill = [Path(command[command.index(option) + 1]) for option in ('--decode-so', '--prefill-so')]
        cell = dict(model='qwen3_1_7b', model_dir='/root/models/qwen3_1_7b_rand', batch=batch, tp=1, moe=False,
            decode=ref(decode), prefill=ref(prefill), generated_source=ref(str(decode) + '.cu'),
            tokens=ref(directory / f'candidate{index}/round0/measure/tokens_N1024_run1.json'),
            tpot_ms=candidate['measured_tpot_ns']/1e6,
            prompt=str(ROOT / 'configs/prompts/qwen_isl64.json'), reference_selection=ref(directory / 'selection.json'))
        (out / f'qwen3_1_7b_B{batch}.json').write_text(json.dumps(cell, indent=2) + '\n')
    for row in csv.DictReader((ROOT / 'docs/results/MX6_final_moe_c550.csv').open()):
        directory = ROOT / row['raw_directory']
        files = [p for p in directory.rglob('plan.so') if p.parent.name == 'decode' and sha(p) == row['binary_sha256']]
        if len(files) != 1:
            raise ValueError('cannot resolve final MoE binary: ' + str(directory))
        decode = files[0]
        prefill = decode.parent.parent / 'prefill/plan.so'
        if sha(prefill) != row['prefill_binary_sha256']:
            raise ValueError('frozen MoE prefill identity differs')
        reference = decode.parent.parent / 'correctness/check/c2/tokens_L1_loop.json'
        if not reference.exists():
            matches = sorted(directory.rglob('tokens_N1024_run1.json'))
            reference = matches[0]
        model = row['model']
        cell = dict(model=model, model_dir='/root/models/' + model + '_rand', batch=int(row['batch']), tp=1, moe=True,
            decode=ref(decode), prefill=ref(prefill), generated_source=ref(str(decode) + '.cu'), tokens=ref(reference),
            tpot_ms=float(row['tpot_ms']), prompt=str(ROOT / ('configs/prompts/olmoe_1b_7b_isl64.json'
                if model.startswith('olmoe') else 'configs/prompts/qwen_isl64.json')),
            reference_selection=ref(ROOT / 'docs/results/MX6_final_moe_c550.csv'), canonical_pin=row['canonical_pin'])
        (out / f'{model}_B{row["batch"]}.json').write_text(json.dumps(cell, indent=2) + '\n')
    print('resolved', len(list(out.glob('*.json'))), 'references (nine ablation cells plus MoE routing scopes); no AB4b jobs')


if __name__ == '__main__':
    main()
