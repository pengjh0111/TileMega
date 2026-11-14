"""Validate P-CORR output determinism independently of reference metrics."""
import math
import re


def validate_bert_checks(checks, artifact_id):
    if len(checks) != 5:
        raise ValueError('BERT P-CORR requires five fresh checks')
    required = {'reference', 'synthetic_weights', 'batch', 'checker_sha256',
                'artifact_id', 'bridge_sha256', 'archive_sha256',
                'source_checkpoint_sha256', 'input_tensors_sha256', 'passed'}
    first = checks[0]
    metadata = {k: v for k, v in first.items() if k != 'cases'}
    if (not required <= metadata.keys() or metadata['passed'] is not True or
            metadata['synthetic_weights'] is not False or
            metadata['reference'] != 'exported BF16 checkpoint promoted to FP32' or
            metadata['artifact_id'] != artifact_id or metadata['batch'] not in (1, 2, 4, 8, 16)):
        raise ValueError('BERT reference or artifact metadata is incomplete')
    signature = None
    metrics = {}
    for check in checks:
        if {k: v for k, v in check.items() if k != 'cases'} != metadata:
            raise ValueError('BERT check metadata changed across processes')
        cases = check.get('cases', [])
        if [(c.get('epoch'), c.get('mode')) for c in cases] != [
                (epoch, mode) for epoch in range(3) for mode in (1, 2)]:
            raise ValueError('BERT check lacks the six execution cases')
        for case in cases:
            if set(case) != {'epoch', 'mode', 'outputs'}:
                raise ValueError('unexpected BERT execution case')
            outputs = case['outputs']
            if (len(outputs) != 2 or
                    [o.get('name') for o in outputs] != ['layer_norm_24', 'tanh']):
                raise ValueError('BERT check lacks its two model outputs')
            current = []
            for output in outputs:
                if set(output) != {'name', 'cosine_min', 'max_error',
                                   'bf16_max_error', 'output_sha256'}:
                    raise ValueError('unexpected BERT output metric schema')
                digest = output['output_sha256']
                if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                    raise ValueError('BERT output lacks its byte hash')
                for field in ('cosine_min', 'max_error', 'bf16_max_error'):
                    value = output[field]
                    if not isinstance(value, (float, int)) or not math.isfinite(value):
                        raise ValueError('BERT output metric is nonfinite')
                    if value < (.999 if field == 'cosine_min' else 0):
                        raise ValueError('BERT output fails the unchanged FP32 criterion')
                    metrics.setdefault((output['name'], field), []).append(value)
                current.append((output['name'], digest))
            if signature is None:
                signature = current
            if current != signature:
                raise ValueError('BERT output bytes differ across cases or processes')
    return {'cases': first['cases'], 'metric_ranges': [
        {'output': name, 'metric': field, 'min': min(values), 'max': max(values)}
        for (name, field), values in sorted(metrics.items())]}


def validate_bert_output_identity(checks, reference, artifact_id):
    result = validate_bert_checks(checks, artifact_id)
    fields = ('reference', 'synthetic_weights', 'batch', 'checker_sha256',
              'bridge_sha256', 'archive_sha256', 'source_checkpoint_sha256',
              'input_tensors_sha256')
    if reference.get('passed') is not True or any(
            checks[0].get(field) != reference.get(field) for field in fields):
        raise ValueError('BERT byte comparison changed its reference inputs or checker')

    def signature(cases):
        return [(case.get('epoch'), case.get('mode'),
                 [(output.get('name'), output.get('output_sha256'))
                  for output in case.get('outputs', [])]) for case in cases]

    # Hidden states and the pooler are different tensors. Match each named
    # output in every execution case, rather than collapsing their hash sets.
    if signature(result['cases']) != signature(reference.get('cases', [])):
        raise ValueError('BERT named output bytes differ from the reference')
    return {'passed': True, 'named_output_sha256': {
        output['name']: output['output_sha256']
        for output in result['cases'][0]['outputs']}}
