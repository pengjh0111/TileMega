"""Admit actual native fits, not fabricated legacy coefficients or stale stamps."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from tilemega.cli import missing_serving_fit_sections, ATTENTION_FIT_SHAPES

gemms = ('gemm_store', 'gemm_residual', 'gemm_swiglu', 'gemm_argmax_partial')
kinds = (*gemms, 'embedding', 'rmsnorm', 'argmax_reduce', 'attention_merge',
         *(f'fused_attention_{phase}_d{d}_q{q}_norm{norm}'
           for d,q,norm in ATTENTION_FIT_SHAPES for phase in ('decode','prefill')))
body = dict(samples=0, source='', fixed=[], loop_body=[], loop_wait=[], loop_fixed=[],
    serving={k: dict(fixed_ns=100., byte_ns=.01, flop_ns=.001, samples=4) for k in kinds})
body['serving_paged'] = {f'{g}_n{n}_k{k}': dict(fixed_ns=100., iter_ns=20., median_relative_error=.01)
    for g in gemms for n, k in ((128, 64), (32, 64), (64, 128), (64, 64))}
for d,q,norm in ATTENTION_FIT_SHAPES:
    body['serving_paged'][f'attention_decode_d{d}_q{q}_norm{norm}'] = dict(fixed_ns=100., iter_ns=20., median_relative_error=.01)
body['serving_paged']['loader'] = dict(stream_gbps_per_sm=10., aggregate_gbps=1000.)
target = dict(calibration_by_dtype=dict(bf16=dict(task_body=body)))
assert missing_serving_fit_sections(target) == []
assert set(missing_serving_fit_sections({})) == {'task_bodies', 'task_bodies_paged'}
bad = copy.deepcopy(target)
del bad['calibration_by_dtype']['bf16']['task_body']['serving']['embedding']
assert missing_serving_fit_sections(bad) == ['task_bodies']
# No Q/norm combination may silently borrow the old D-only coefficient.
for d,q,norm in ATTENTION_FIT_SHAPES:
    bad=copy.deepcopy(target)
    del bad['calibration_by_dtype']['bf16']['task_body']['serving'][f'fused_attention_decode_d{d}_q{q}_norm{norm}']
    assert missing_serving_fit_sections(bad)==['task_bodies']
    bad=copy.deepcopy(target)
    del bad['calibration_by_dtype']['bf16']['task_body']['serving_paged'][f'attention_decode_d{d}_q{q}_norm{norm}']
    assert missing_serving_fit_sections(bad)==['task_bodies_paged']
for value in (-1., float('inf'), float('nan'), True):
    bad = copy.deepcopy(target)
    bad['calibration_by_dtype']['bf16']['task_body']['serving_paged']['gemm_store_n128_k64']['iter_ns'] = value
    assert missing_serving_fit_sections(bad) == ['task_bodies_paged']
print('PASS native serving-only fits; missing/nonfinite/negative fits invalidate section cache')
