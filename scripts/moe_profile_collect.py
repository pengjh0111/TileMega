"""Publish compact accepted timings and explicit matrix completion states."""
import argparse
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=ROOT / 'runs/moe_profile/benchmark')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/results/moe_profile')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT / 'configs/moe_profile/workload.json')
    rows = []
    for framework in plan['frameworks']:
        for length, output in plan['lengths']:
            for batch in plan['batches']:
                cell = f'{framework}_tp1_b{batch}_i{length}_o{output}'
                attempts = sorted((args.raw / cell).glob('attempt*'))
                accepted = [attempt for attempt in attempts
                            if read(attempt / 'guard/guard_result.json').get('code') == 0
                            and read(attempt / 'worker/result.json').get('status') == 'ok']
                attempt = accepted[-1] if accepted else attempts[-1] if attempts else None
                worker = read(attempt / 'worker/result.json') if attempt else {}
                guard = read(attempt / 'guard/guard_result.json') if attempt else {}
                summary = worker.get('summary', {}) if accepted else {}
                telemetry_path = attempt / 'guard/guard.jsonl' if attempt else None
                telemetry = ([json.loads(line) for line in telemetry_path.read_text().splitlines()]
                             if telemetry_path and telemetry_path.exists() else [])
                running = [row for row in telemetry if row.get('phase') == 'running']
                engine = read(attempt / 'worker/engine.json') if attempt else {}
                modules = engine.get('worker_model', [])
                if isinstance(modules, dict):
                    modules = [modules]
                attention = sorted({item['impl'] for rank in modules
                                    for item in rank.get('module_types', []) if 'impl' in item})
                moe = sorted({item['quant_method'] for rank in modules
                              for item in rank.get('module_types', [])
                              if 'experts' in item.get('example_name', '') and 'quant_method' in item})
                if framework == 'sglang':
                    server = engine.get('server_info', {})
                    attention = [str(server.get('attention_backend', ''))]
                    moe = [str(server.get('moe_runner_backend', ''))]
                if framework == 'trtllm' and not moe:
                    moe = sorted({item['cls'] for rank in modules for item in rank.get('module_types', [])
                                  if item['cls'].endswith(('CutlassFusedMoE', 'TRTLLMGenFusedMoE'))})
                round_files = sorted((attempt / 'worker').glob('round[0-9].json')) if attempt else []
                round_rows = [read(path) for path in round_files]
                before = round_rows[0].get('before_telemetry', {}) if round_rows else {}
                after = round_rows[-1].get('after_telemetry', {}) if round_rows else {}
                native = [item for round_row in round_rows
                          for item in round_row.get('engine_metrics', {}).values() if isinstance(item, dict)]
                def native_median(name):
                    values = [item[name] for item in native if isinstance(item.get(name), (float, int))]
                    return statistics.median(values) if values else ''
                allocator = worker.get('allocator_peak', [])
                if isinstance(allocator, dict):
                    allocator = [allocator]
                memory = worker.get('final_server_info', {}).get('internal_states', [{}])[0].get('memory_usage', {})
                row = dict(framework=framework, tensor_parallel_size=1, batch=batch,
                           input_tokens=length, output_tokens=output,
                           status='ok' if accepted else 'pending' if not attempt else
                                  'deferred_or_invalid' if guard.get('code') == 75 else
                                  'running' if not guard else 'failed',
                           reason='' if accepted else worker.get('error') or guard.get('reason', ''),
                           rounds=summary.get('rounds', ''),
                           ttft_median_ms=summary.get('ttft_request_median_ms', ''),
                           ttft_round_range_ms=summary.get('ttft_request_median_ms_range', ''),
                           tpot_median_ms=summary.get('tpot_request_median_ms', ''),
                           tpot_round_range_ms=summary.get('tpot_request_median_ms_range', ''),
                           e2e_median_ms=summary.get('e2e_ms', ''),
                           e2e_round_range_ms=summary.get('e2e_ms_range', ''),
                           output_tokens_per_s=summary.get('output_tokens_per_s', ''),
                           throughput_round_range=summary.get('output_tokens_per_s_range', ''),
                           actual_input_tokens=summary.get('actual_input_tokens', ''),
                           actual_output_tokens=summary.get('actual_output_tokens', ''),
                           output_hash_stable=summary.get('output_hash_stable', ''),
                           gpu=worker.get('cuda_visible_devices', ''),
                           gpu_uuid=telemetry[0].get('gpu_uuid', '') if telemetry else '',
                           start_sm_mhz=before.get('sm_mhz', ''), end_sm_mhz=after.get('sm_mhz', ''),
                           boundary_clock_status='exact round boundary NVML samples' if before and after else
                                                 'unavailable at round boundary;guard samples retained',
                           running_sm_min_mhz=min((item['sm_clock_mhz'] for item in running), default=''),
                           running_sm_max_mhz=max((item['sm_clock_mhz'] for item in running), default=''),
                           peak_nvml_sampled_mib=max((row['used_mib'] for row in running), default=''),
                           memory_scope='1s NVML sampled maximum across loading/warmup/timing;'
                                        'may miss transient peaks;allocator peaks separately in worker result',
                           torch_allocator_peak_allocated_bytes=max((item['max_allocated_bytes'] for item in allocator), default=''),
                           torch_allocator_peak_reserved_bytes=max((item['max_reserved_bytes'] for item in allocator), default=''),
                           kv_capacity_tokens=memory.get('token_capacity', ''),
                           kv_allocated_gib=memory.get('kvcache', ''),
                           kv_precision='BF16',
                           kv_usage_status='Capacity/config and actual sampled Decode KV lengths retained;no peak active KV counter',
                           native_prefill_latency_median_s=native_median('prefill_latency'),
                           native_queue_time_median_s=native_median('queue_time'),
                           native_forward_time_median_s=native_median('forward_time'),
                           native_cached_tokens_max=max((item['cached_tokens'] for item in native if 'cached_tokens' in item), default=''),
                           native_total_retractions_max=max((item['total_retractions'] for item in native if 'total_retractions' in item), default=''),
                           native_metric_scope='SGLang SDK seconds,request aggregation across measured rounds;'
                                               'forward_time includes multiple forwards,not GPU-only Prefill/Decode. '
                                               'vLLM/TRTLLM per-request engine timers unavailable in installed SDK.',
                           attention_backend=';'.join(attention), moe_backend=';'.join(moe),
                           backend_evidence=str(attempt / 'worker/engine.json') if attempt else '',
                           model_revision=plan['model']['revision'],
                           evidence='verified' if accepted else 'no accepted performance measurement',
                           metric_scope='Host-observed closed-cohort SDK streaming;TPOT=(last-first)/(O-1)',
                           repeat_scope='Three rounds in the same warmed engine process',
                           full_resident_batch_fit='no' if batch == 64 and length == 8192 else 'not_excluded_by_weight_KV_lower_bound',
                           capacity_scope='BF16 weights plus 98304 KV bytes/token exceed 80GiB at B64 8192+256. '
                                          'Any completed run is a closed cohort with engine admission/preemption,not resident fixedB64.'
                                          if batch == 64 and length == 8192 else '',
                           raw_evidence=str(attempt) if attempt else '')
                rows.append(row)
    with (args.out / 'benchmark.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    states = {state: sum(row['status'] == state for row in rows) for state in sorted({row['status'] for row in rows})}
    print(json.dumps(dict(cells=len(rows), status=states)))


if __name__ == '__main__':
    main()
