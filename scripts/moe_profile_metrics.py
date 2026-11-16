"""Metric definitions shared by the baseline workers and their CPU tests."""
import hashlib
import json
import math
import statistics


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def summarize_round(batch, input_length, output_length, start_ns, end_ns, events):
    if batch < 1 or input_length < 1 or output_length < 2 or end_ns <= start_ns:
        raise ValueError('invalid closed-cohort dimensions or clock interval')
    requests = [dict(first=None, last=None, ids=[], chunks=[]) for _ in range(batch)]
    previous = start_ns
    for event in events:
        index, now, ids = event['index'], event['observed_ns'], event['ids']
        if not 0 <= index < batch or not start_ns <= now <= end_ns or now < previous:
            raise ValueError('invalid event index or monotonic timestamp')
        previous = now
        request = requests[index]
        if ids[:len(request['ids'])] != request['ids']:
            raise ValueError('stream lost its cumulative token prefix')
        delta = len(ids) - len(request['ids'])
        if delta:
            request['first'] = now if request['first'] is None else request['first']
            request['last'] = now
            request['chunks'].append(delta)
            request['ids'] = ids
    if any(len(row['ids']) != output_length or row['first'] is None for row in requests):
        raise ValueError('actual output counts do not match the declared fixed cohort')
    ttft = [(row['first'] - start_ns) / 1e6 for row in requests]
    tpot = [(row['last'] - row['first']) / (output_length - 1) / 1e6 for row in requests]
    elapsed_ms = (end_ns - start_ns) / 1e6
    return dict(batch=batch, input_tokens_per_request=input_length,
                output_tokens_per_request=output_length, actual_input_tokens=batch*input_length,
                actual_output_tokens=batch*output_length, requests=batch,
                e2e_ms=elapsed_ms, ttft_request_median_ms=statistics.median(ttft),
                ttft_request_max_ms=max(ttft), tpot_request_median_ms=statistics.median(tpot),
                output_tokens_per_s=batch*output_length/(elapsed_ms/1000),
                ttft_per_request_ms=ttft, tpot_per_request_ms=tpot,
                output_hashes=[token_hash(row['ids']) for row in requests],
                streamed_chunk_sizes=[row['chunks'] for row in requests],
                all_chunks_single_token=all(size == 1 for row in requests for size in row['chunks']),
                metric_scope='Host-observed streamed request latency and closed-cohort throughput; '
                             'TPOT divides by actual output_tokens-1;not GPU iteration time/utilization.')


def interval_union(intervals):
    """Count overlapping GPU activity once, independently of summed kernel time."""
    total, edge = 0, None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('negative activity interval')
        total += max(0, end - max(start, edge if edge is not None else start))
        edge = max(end, edge if edge is not None else end)
    return total


def summarize_repeats(rows):
    if len(rows) < 3:
        raise ValueError('formal baseline requires at least three complete measured rounds')
    keys = ('batch', 'input_tokens_per_request', 'output_tokens_per_request',
            'actual_input_tokens', 'actual_output_tokens', 'requests')
    if any(any(row[key] != rows[0][key] for key in keys) for row in rows):
        raise ValueError('repeats changed the input/output matrix')
    result = dict(rounds=len(rows), **{key: rows[0][key] for key in keys})
    for key in ('e2e_ms', 'ttft_request_median_ms', 'ttft_request_max_ms',
                'tpot_request_median_ms', 'output_tokens_per_s'):
        values = [row[key] for row in rows]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError('formal latency/throughput lacks finite positive measurements')
        result[key] = statistics.median(values)
        result[key + '_range'] = max(values) - min(values)
    result['output_hash_stable'] = all(row['output_hashes'] == rows[0]['output_hashes'] for row in rows)
    return result
