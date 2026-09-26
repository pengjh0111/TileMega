"""Check the units and completeness rules of the R11 page-chain report."""
from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'docs/experiments/SERVING_R11'))
from analyze_page_chain import analyze
from archive_page_vector_diagnostics import SUMMARY


def write(path: Path, rows: list[dict]) -> None:
    with path.open('w') as stream:
        table = csv.DictWriter(stream, fieldnames=rows[0], delimiter='\t')
        table.writeheader()
        table.writerows(rows)


with tempfile.TemporaryDirectory() as temporary:
    folder = Path(temporary)
    trace, floor, chain = (folder / name for name in ('trace.tsv', 'floor.tsv', 'chain.tsv'))
    write(trace, [
        dict(step=0, worker=0, kernel_begin_ns=100, kernel_end_ns=300,
             dependency_wait_ns=40, page_full_ns=30, full_and_wait_ns=10),
        dict(step=0, worker=1, kernel_begin_ns=110, kernel_end_ns=290,
             dependency_wait_ns=50, page_full_ns=40, full_and_wait_ns=20),
        dict(step=1, worker=0, kernel_begin_ns=320, kernel_end_ns=560,
             dependency_wait_ns=60, page_full_ns=50, full_and_wait_ns=30),
        dict(step=1, worker=1, kernel_begin_ns=330, kernel_end_ns=550,
             dependency_wait_ns=70, page_full_ns=60, full_and_wait_ns=40),
    ])
    write(floor, [dict(past=64, floor_ns=100), dict(past=65, floor_ns=120)])
    write(chain, [dict(cp_nodes=2, cp_chain_span_ns=200)])
    samples, report = analyze(trace, model='llama', batch=1, prompt_len=64,
                              floor_points=floor, chain_analysis=chain,
                              chain_past=64)
    assert len(samples) == report['steps'] == 2
    assert set(SUMMARY).issubset(report)
    assert [row['kernel_span_ns'] for row in samples] == [200, 240]
    assert [row['residual_bubble_ns_per_link'] for row in samples] == [50, 60]
    assert samples[1]['launch_gap_ns'] == report['launch_gap_p50_ns'] == 20
    assert report['launch_gap_total_ns'] == 20
    assert report['launch_gap_p90_ns'] == report['launch_gap_p99_ns'] == 20
    assert report['launch_gap_fraction_of_decode_span'] == 20 / 460
    assert report['measured_chain_over_floor'] == 2
    assert report['measured_chain_excess_over_floor_ns'] == 100
    assert report['measured_chain_residual_bubble_ns_per_link'] == 50
    assert report['page_full_and_dependency_wait_cta_ns'] == 100
    assert report['page_full_and_dependency_wait_mean_cta_ns_per_step'] == 25
    with trace.open() as stream:
        invalid = list(csv.DictReader(stream, delimiter='\t'))
    invalid[0]['full_and_wait_ns'] = '31'  # Exceeds that CTA's 30 ns page-full marginal.
    write(trace, invalid)
    try:
        analyze(trace, model='llama', batch=1, prompt_len=64,
                floor_points=floor, chain_analysis=chain, chain_past=64)
    except ValueError as error:
        assert 'page-wait counters exceed CTA time' in str(error)
    else:
        raise AssertionError('invalid CTA-local overlap was accepted')
    print('PASS page-chain exact floors, realized bubbles, CTA overlap and launch gap')
