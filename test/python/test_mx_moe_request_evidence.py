"""CPU replay of actual MoE request evidence, including corrupted summaries."""
import copy
import json
from pathlib import Path

import pytest

from tilemega.moe.request_evidence import validate_report, validate_request

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'runs/mx5/moe_rank_whole_program_candidate_v1'
pytestmark = pytest.mark.skipif(not RAW.is_dir(), reason='requires existing manual MoE measurements; no GPU')


def inputs(batch=1, arm='candidate', round=1):
    base = RAW if arm == 'candidate' else ROOT / 'runs/mx5/moe_native_cli_cpu_v1'
    if arm == 'candidate':
        binaries = [base / ('B%d_w0_p%d' % (batch, prompt)) / 'plan.so' for prompt in (0, 1)]
    else:
        binaries = [base / ('B%d_p%d' % (batch, prompt)) / 'plan.so' for prompt in (0, 1)]
    report = RAW / ('PPAIR_B%d' % batch) / ('%s_round%d' % (arm, round)) / 'measure/measurements.json'
    return report, *binaries


@pytest.mark.parametrize('batch', (1, 16))
@pytest.mark.parametrize('arm', ('baseline', 'candidate'))
@pytest.mark.parametrize('round', (1, 2))
def test_actual_request_replay_preserves_protocol_and_diagnostic_boundary(batch, arm, round):
    report, decode, prefill = inputs(batch, arm, round)
    result = validate_request(report, decode, prefill, batch)
    assert result['tpot_ns'] == json.loads(report.read_text())['tpot_seconds'] * 1e9
    assert result['plan_pair_contract']['shared_binding_contract']
    assert all(result[k] is False for k in ('model_acceptance_implied', 'grants_512_exception',
        'fresh_process_or_guard_qualification_implied', 'final', 'default_enabled'))


@pytest.mark.parametrize('change', ('summary', 'ttft', 'tokens', 'shape', 'token_type', 'vocabulary',
    'throughput', 'nan_latency', 'watchdog', 'diag', 'frame', 'prefill_executor',
    'prefill_loop', 'artifact', 'run_count', 'formula', 'synthetic', 'kernel',
    'execution_digest', 'shared_memory', 'run_kernel', 'run_coordinate_type', 'boolean_batch'))
def test_edited_summary_tokens_or_artifact_is_rejected(change):
    report, decode, prefill = inputs()
    data = json.loads(report.read_text())
    identities = [json.loads(Path(str(p) + '.identity.json').read_text()) for p in (decode, prefill)]
    if change in ('summary', 'ttft'):
        key = 'e2e_seconds' if change == 'summary' else 'ttft_seconds'
        data[key] += .01
        data['tpot_seconds'] = (data['e2e_seconds'] - data['ttft_seconds']) / 1023
    elif change in ('tokens', 'shape', 'token_type', 'vocabulary'):
        run = next(r for r in data['runs'] if r['N'] == 1024 and r['run'] == 2)
        if change == 'shape':
            run['tokens'][0].pop()
        else:
            run['tokens'][0][0] = dict(tokens=1, token_type=True, vocabulary=151936)[change]
    elif change == 'throughput':
        data['output_tokens_per_second'] *= 2
    elif change == 'nan_latency':
        data['runs'][0]['e2e_seconds'] = float('nan')
    elif change in ('watchdog', 'diag'):
        identities[1]['plan'][change] = 1
    elif change == 'frame':
        next(iter(identities[1]['resources'].values()))['stack_bytes'] = 16
    elif change in ('prefill_executor', 'prefill_loop'):
        key, value = ('executor', 'L2') if change == 'prefill_executor' else ('loop', True)
        data['prefill_execution_identity'][key] = value
    elif change == 'artifact':
        data['execution_identity']['artifact_id'] = '0' * 64
    elif change == 'run_count':
        data['runs'].pop()
    elif change == 'formula':
        data['tpot_seconds'] += .001
    elif change in ('kernel', 'execution_digest', 'shared_memory'):
        key, value = dict(kernel=('kernel', 'wrong_kernel'), execution_digest=('execution_id', '0' * 64),
            shared_memory=('shared_memory_bytes', 32784))[change]
        data['prefill_execution_identity'][key] = value
    elif change == 'run_kernel':
        data['runs'][0]['execution_identity']['kernel'] = 'wrong_kernel'
    elif change == 'run_coordinate_type':
        data['runs'][0]['run'] = False
    elif change == 'boolean_batch':
        with pytest.raises(ValueError):
            validate_report(data, *identities, True)
        return
    else:
        data['synthetic_fixture'] = True
    with pytest.raises(ValueError):
        validate_report(copy.deepcopy(data), *identities, 1)
