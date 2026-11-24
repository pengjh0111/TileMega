"""Verify artifact-bound same-binary C1 and C2 evidence for A800 R11."""
import json
from pathlib import Path

from a100_baselines import sha
from tilemega.serving.execution import advertised_execution_combinations


GATE = 'sm90-c1-c2-v1'
INVARIANTS = ('gpu_uuid', 'checkpoint_sha256', 'config_sha256', 'prompt_sha256')


def read(path):
    return json.loads(Path(path).read_text())


def executor_agreement(c2):
    expected = set(advertised_execution_combinations(c2['supported_modes'], c2['supported_loop_modes']))
    rows = c2['checks']
    checked = {(row['mode'], row['decode_loop']) for row in rows if row['pass'] is True
               and row['decode_loop_used'] == bool(row['decode_loop'])}
    return (c2['pass'] is True and c2['same_binary'] is True and len(rows) == len(expected)
            and checked == expected)


def verify(path, *, job, model, batch, identity):
    receipt = read(path)
    if (receipt['job'] != str(job) or (receipt['model'], receipt['batch']) != (model, batch)
            or receipt['stage'] != 'correctness' or receipt['numeric_gate'] != GATE
            or not all(receipt[name] is True for name in ('pass', 'hf_pass', 'executor_agree'))
            or any(receipt[name] != digest for name, digest in identity.items())):
        raise ValueError('C1/C2 receipt differs from the frozen artifact pair')
    evidence = Path(receipt['evidence'])
    sample = evidence/f'B{batch}'/'measurements.json'
    result = read(sample)
    reference, c1_path, c2_path = (evidence/name for name in ('reference_tokens.json','c1.json','c2.json'))
    if (sha(reference) != receipt['tokens_sha256'] or sha(sample) != receipt['process_receipt_sha256']
            or sha(c1_path) != receipt['hf_proof_sha256'] or sha(c2_path) != receipt['executor_proof_sha256']
            or result['acceptance'] != 'strict' or not result['ownership_verified']
            or result['configuration']['decode_sha256'] != identity['decode_sha256']
            or result['configuration']['prefill_sha256'] != (
                'prompt mode' if identity['prefill_sha256']=='prompt' else identity['prefill_sha256'])
            or result['checkpoint']['checkpoint_sha256'] != result['configuration']['checkpoint_sha256']):
        raise ValueError('invalid guarded C1/C2 process provenance')
    c1, c2 = read(c1_path), read(c2_path)
    positions = 256 if model == 'qwen3_30b_a3b' else 1024
    if (c1['pass'] is not True or c1['positions'] != positions*batch
            or receipt['hf_positions'] != positions or not executor_agreement(c2)):
        raise ValueError('C1/C2 uses the wrong teacher-forced window or executor order')
    return dict(receipt=receipt, path=str(path), sha256=sha(path), pid=result['pid'],
                gpu_uuid=receipt['device_uuid'], checkpoint_sha256=result['checkpoint']['checkpoint_sha256'],
                config_sha256=result['model_config_sha256'], prompt_sha256=result['prompt_ids_sha256'])
