#!/usr/bin/env python3
"""Compare two independently validated cohorts for one A800 model and batch."""
import argparse
import json
from pathlib import Path

from a100_baselines import BATCHES, MODELS, inspect_inputs, sha
from a800_baselines import INVENTORY, RUN, hardware_class
from a800_results import native_cell


def reference_stream(model, batch, arm, device):
    cell = f'*/{model}/B{batch}'
    for pattern in ('*/correctness-*.json', 'correctness-*.json'):
        for path in sorted((RUN/'native').glob(f'{cell}/{pattern}')):
            check = json.loads(path.read_text())
            if not check.get('pass') or not check.get('hf_pass') or any(check[k] != arm[k] for k in
                    ('prefill_sha256', 'decode_sha256', 'target_sha256')):
                continue
            if ('device' in check and hardware_class(check['device']) != hardware_class(device)) or (
                    'device' not in check and check['device_uuid'] != device['uuid']):
                continue
            root = Path(check['evidence'])
            tokens = root/'reference_tokens.json'
            values = json.loads(tokens.read_text())
            hf = json.loads((root/'c1.json').read_text())
            if (sha(tokens) != check['tokens_sha256'] or len(values) != batch or any(len(seq) != 1024 for seq in values)
                    or not hf['pass'] or hf['batch'] != batch or hf['max_new_tokens'] != 1024):
                raise ValueError('invalid full-length teacher-forced evidence')
            arm.update(tokens_sha256=check['tokens_sha256'], hf_max_gap=hf['max_gap'],
                       hf_receipt_sha256=sha(root/'c1.json'))
            return values
    raise ValueError('could not find the full-length HF evidence for this artifact')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch', type=int, choices=BATCHES, required=True)
    parser.add_argument('--device-target', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--candidate-plan', required=True, type=Path)
    args = parser.parse_args()

    plan = json.loads(args.candidate_plan.read_text())
    if ((plan['model'], plan['batch']) != (args.model, args.batch)
            or plan['stage'] != 'candidate-timing'):
        parser.error('require the independent timing plan for this candidate and cell')
    fields = ('prefill_sha256', 'decode_sha256', 'target_sha256')
    selections = json.loads(args.selection.read_text())
    control = selections.get(f'{args.model}/B{args.batch}')
    inputs = dict(device=json.loads(args.device_target.read_text())['device_provenance'],
                  models=inspect_inputs(INVENTORY))
    arms = {}
    weights = set()
    for name, triple in (('control', control), ('candidate', plan)):
        selected = tuple(triple[k] for k in fields) if triple is not None else None
        rows, cell = native_cell(inputs, args.model, args.batch, selected)
        if (cell['status'] != 'measured' or cell['accepted_correctness_processes'] < 5
                or len(rows) != cell['required_processes']):
            parser.error(f'{name} lacks complete timing and correctness')
        weights.update(row['checkpoint_sha256'] for row in rows)
        arms[name] = {k: cell[k] for k in (*fields, 't1_s', 't1024_s', 'tpot_ms', 'processes',
                                            'correctness', 'config_sha256', 'prompt_sha256', 'receipt_sha256')}
    if len(weights) != 1:
        parser.error('the two arms used different checkpoint weights')
    old, new = arms['control'], arms['candidate']
    if old['decode_sha256'] == new['decode_sha256']:
        parser.error('the candidate is already the selected decode artifact')
    if any(old[key] != new[key] for key in ('prefill_sha256', 'config_sha256', 'prompt_sha256')):
        parser.error('the decode comparison changed its prefill, configuration or prompt')
    control_tokens, candidate_tokens = (reference_stream(args.model, args.batch, arm, inputs['device'])
                                        for arm in (old, new))
    differences = [[i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]
                   for a, b in zip(control_tokens, candidate_tokens)]
    print(json.dumps(dict(model=args.model, batch=args.batch, checkpoint_sha256=weights.pop(), arms=arms,
        differing_tokens=sum(map(len, differences)), first_difference_by_sequence=[next(iter(d), None) for d in differences],
        lower_tpot=new['tpot_ms'] < old['tpot_ms'], lower_t1024=new['t1024_s'] < old['t1024_s']),
        indent=2))


if __name__ == '__main__':
    main()
