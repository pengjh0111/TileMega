"""End-to-end summaries; component timing never selects defaults."""
import statistics


def metrics(record, workload, *, step_bytes=None, dram_gbps=None):
    batch, osl = int(workload['batch']), int(workload['osl'])
    e2e, ttft = float(record['e2e_seconds']), float(record['ttft_seconds'])
    if osl <= 1 or e2e <= ttft or ttft < 0:
        raise ValueError('invalid end-to-end measurement')
    tpot = (e2e - ttft) / (osl - 1)
    return {'tpot_seconds': tpot, 'request_seconds': e2e,
            'ttft_seconds': ttft, 'output_tokens_per_second': batch * osl / e2e,
            'mbu': (step_bytes / (tpot * dram_gbps * 1e9)
                    if step_bytes is not None and dram_gbps else None)}


def summarize(values):
    if not values:
        raise ValueError('no accepted measurements')
    return {'median': statistics.median(values), 'range': max(values) - min(values),
            'samples': list(values)}


def discernible(left, right):
    a, b = summarize(left), summarize(right)
    delta = abs(a['median'] - b['median'])
    return delta > max(a['range'], b['range'], .005 * b['median'])
