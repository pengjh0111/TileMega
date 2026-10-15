#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Make a scoped DNN A/B recommendation from validated full-model evidence."""
import argparse
import json
from pathlib import Path
import statistics

from r16_measurement_validation import file_sha, native_identity
from summarize_r16 import collect_dnn


def comparison(reference, candidate):
    result = {}
    for field in ('median_ms', 'p90_ms'):
        a = [float(row[field]) for row in reference]
        b = [float(row[field]) for row in candidate]
        center_a, center_b = statistics.median(a), statistics.median(b)
        threshold = max(.005 * center_a, max(a)-min(a), max(b)-min(b))
        delta = center_b-center_a
        result[field] = dict(reference=center_a, candidate=center_b,
            reference_samples=a, candidate_samples=b, indistinguishable_ms=threshold,
            relative_change=delta/center_a,
            verdict='indistinguishable' if abs(delta) <= threshold else
                ('faster' if delta < 0 else 'slower'))
    verdicts = {row['verdict'] for row in result.values()}
    return dict(metrics=result, eligible_in_measured_grid='slower' not in verdicts and 'faster' in verdicts)


def evidence(config_path, records):
    config = json.loads(config_path.read_text())
    directory = Path(config['directory'])
    identity = native_identity(directory/'plan.so')
    gate_path = directory/'gate-5fresh.json'
    gate = json.loads(gate_path.read_text())
    checkpoint = json.loads((Path(config['export'])/'manifest.json').read_text())['checkpoint']['sha256']
    if file_sha(Path(config['export'])/'checkpoint/model.safetensors') != checkpoint:
        raise ValueError('source checkpoint differs from export manifest')
    if not gate.get('passed') or gate.get('fresh_passes') != 5 or gate.get('fresh_processes') != 5 or len(set(gate['fresh_child_pids'])) != 5:
        raise ValueError('five fresh complete-model numerical checks are required')
    if gate['artifact_id'] != identity['artifact_id'] or gate['source_checkpoint_sha256'] != checkpoint:
        raise ValueError('gate belongs to a different binary or checkpoint')
    for path in gate['fresh_receipts']:
        item = json.loads(Path(path).read_text())
        if not item.get('passed') or any(item[key] != gate[key] for key in
                ('artifact_id', 'archive_sha256', 'input_tensors_sha256')):
            raise ValueError('fresh numerical receipt differs from aggregate')
        if {row['mode'] for row in item['cases']} != {1,2}:
            raise ValueError('same-binary L1/L2 numerical checks are absent')
    rows = [row for row in records if row['candidate_id'] == config['candidate_id']]
    if {row['round'] for row in rows} != {0,1,2} or len(rows) != 3 or len({row['pid'] for row in rows}) != 3:
        raise ValueError('three distinct completed measurement processes are required')
    if any(row['native_artifact_ids'] != identity['artifact_id'] or
           row['source_checkpoint_sha256'] != checkpoint or
           row['exported_program_sha256'] != gate['archive_sha256'] for row in rows):
        raise ValueError('timed evidence differs from the certified binary/checkpoint')
    return rows, dict(candidate_id=config['candidate_id'], candidate_path=str(config_path),
        candidate_sha256=file_sha(config_path), library=str(directory/'plan.so'),
        artifact_id=identity['artifact_id'], gate_path=str(gate_path), gate_sha256=file_sha(gate_path),
        checkpoint_sha256=checkpoint, archive_sha256=gate['archive_sha256'],
        input_sha256=gate['input_tensors_sha256'], selected_coordinates=config['expected'],
        measurements=[row['measurement'] for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records, rejected = collect_dnn(args.root.resolve())
    a, a_evidence = evidence(args.reference.resolve(), records)
    b, b_evidence = evidence(args.candidate.resolve(), records)
    for key in ('checkpoint_sha256', 'archive_sha256', 'input_sha256'):
        if a_evidence[key] != b_evidence[key]:
            raise ValueError('A/B inputs differ: ' + key)
    if len({row['gpu'] for row in a+b}) != 1 or len({(row['model'],row['batch'],row['precision']) for row in a+b}) != 1:
        raise ValueError('A/B requires the same model, batch, precision and card')
    result = dict(evidence='verified', scope='single complete-model measured grid; not all-model G-PERF',
        model=a[0]['model'], batch=a[0]['batch'], gpu=a[0]['gpu'],
        reference=a_evidence, candidate=b_evidence, **comparison(a,b), default_promoted=False,
        acceptance='strict' if all(row['acceptance']=='strict' for row in a+b) else
            'provisional_container_pid_namespace',
        remaining='final-source confirmation and full affected-grid non-regression before default promotion',
        unrelated_rejected_measurements=rejected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('model','batch','metrics','eligible_in_measured_grid','acceptance')}))


if __name__ == '__main__':
    main()
