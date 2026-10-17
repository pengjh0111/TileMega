#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Estimate logical BF16 decode payload utilization from complete R16 cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def dense_payload(config, batch, isl=64, osl=1024):
    if batch < 1 or isl < 1 or osl < 2:
        raise ValueError('positive batch/ISL and at least two output tokens required')
    if config.get('num_experts', 0) or config.get('num_local_experts', 0):
        return None
    hidden = int(config['hidden_size'])
    layers = int(config['num_hidden_layers'])
    heads = int(config['num_attention_heads'])
    kv_heads = int(config.get('num_key_value_heads', heads))
    dim = int(config.get('head_dim', hidden // heads))
    intermediate = int(config['intermediate_size'])
    vocab = int(config['vocab_size'])
    if min(hidden, layers, heads, kv_heads, dim, intermediate, vocab) <= 0:
        raise ValueError('invalid model dimensions')
    # Qwen's explicit head_dim can make Q width differ from hidden_size.
    per_layer = hidden * (heads + 2 * kv_heads) * dim
    per_layer += hidden * heads * dim + 3 * hidden * intermediate
    weights = 2 * (layers * per_layer + hidden * vocab)
    # After prefill, OSL-1 decode steps attend to ISL+1..ISL+OSL-1 rows.
    average_kv_rows = isl + osl / 2
    kv = 4 * batch * layers * kv_heads * dim * average_kv_rows
    return dict(gemm_weight_bytes=weights, kv_read_bytes=kv,
                average_kv_rows=average_kv_rows, logical_bytes_per_decode=weights + kv)


def moe_payload_bounds(config, batch, isl=64, osl=1024):
    """Bound logical distinct-expert reads; B1 has exactly top-k experts."""
    if (config.get('model_type') != 'qwen3_moe' or config.get('decoder_sparse_step',1) != 1
            or config.get('mlp_only_layers') or config.get('shared_expert_intermediate_size',0)):
        return None
    hidden, layers = int(config['hidden_size']), int(config['num_hidden_layers'])
    heads, kv_heads = int(config['num_attention_heads']), int(config['num_key_value_heads'])
    dim, vocab = int(config['head_dim']), int(config['vocab_size'])
    experts, topk = int(config['num_experts']), int(config['num_experts_per_tok'])
    intermediate = int(config['moe_intermediate_size'])
    if min(hidden,layers,heads,kv_heads,dim,vocab,intermediate,batch,isl) <= 0 or osl < 2 or not 1<=topk<=experts:
        raise ValueError('invalid full MoE dimensions')
    # Include Q/K/V/O, the router and lm_head. Padding and backend-specific
    # redundant expert reads remain outside the logical payload convention.
    dense = 2*(layers*(hidden*(heads+2*kv_heads)*dim + hidden*heads*dim + hidden*experts) + hidden*vocab)
    per_expert = 2*3*hidden*intermediate
    rows = isl+osl/2
    kv = 4*batch*layers*kv_heads*dim*rows
    maximum = min(experts,batch*topk)
    return dict(nonexpert_gemm_weight_bytes=dense,
        expert_weight_bytes_per_layer_per_expert=per_expert,
        distinct_experts_per_layer_min=topk, distinct_experts_per_layer_max=maximum,
        kv_read_bytes=kv, average_kv_rows=rows,
        logical_bytes_per_decode_min=dense+layers*topk*per_expert+kv,
        logical_bytes_per_decode_max=dense+layers*maximum*per_expert+kv)


def bandwidth(receipt):
    rows = [row for row in receipt['rows'] if row['name'] == 'bulk' and row['unit'] == 'GB/s']
    if not receipt.get('passed') or len(rows) != 1 or not rows[0].get('passed'):
        raise ValueError('one accepted cross-card bulk bandwidth row is required')
    values = [float(value) for value in rows[0]['medians']]
    if len(values) != len(rows[0]['gpus']) or not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('invalid bandwidth samples')
    return statistics.mean(values), ';'.join(rows[0]['gpus'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--nominal-gbps', type=float, default=3350)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    measured, cards = bandwidth(json.loads(args.calibration.read_text()))
    if not math.isfinite(args.nominal_gbps) or args.nominal_gbps <= 0:
        parser.error('--nominal-gbps must be finite and positive')
    rows = []
    for cell in csv.DictReader(args.llm.open()):
        if cell['protocol_complete'].lower() != 'true' or int(cell['fresh_processes']) != 3:
            continue
        config_path = args.models / cell['model'] / 'config.json'
        payload = dense_payload(json.loads(config_path.read_text()), int(cell['batch']))
        row = dict(cell, model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                   bandwidth_calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                   measured_bandwidth_gbps=measured, calibration_gpus=cards,
                   nominal_bandwidth_gbps=args.nominal_gbps,
                   mbu_scope='inferred logical BF16 GEMM weights plus KV reads; not physical traffic',
                   mbu_limit='excludes small norm/embedding traffic, writes and system-specific redundant reads',
                   gemm_weight_bytes='', kv_read_bytes='', average_kv_rows='',
                   logical_bytes_per_decode='', mbu_measured='', mbu_nominal='',
                   nonexpert_gemm_weight_bytes='', expert_weight_bytes_per_layer_per_expert='',
                   distinct_experts_per_layer_min='', distinct_experts_per_layer_max='',
                   logical_bytes_per_decode_min='', logical_bytes_per_decode_max='',
                   mbu_measured_min='',mbu_measured_max='',mbu_nominal_min='',mbu_nominal_max='')
        if payload is None:
            bounds = moe_payload_bounds(json.loads(config_path.read_text()), int(cell['batch']))
            row['mbu_scope'] = 'unavailable point estimate: actual per-step distinct MoE expert routing trace required'
            if bounds:
                tpot=float(cell['tpot_ms'])/1000
                if not math.isfinite(tpot) or tpot<=0:raise ValueError('invalid TPOT')
                row.update(bounds)
                for name,value in [('measured',measured),('nominal',args.nominal_gbps)]:
                    for end in ('min','max'):
                        row[f'mbu_{name}_{end}']=bounds['logical_bytes_per_decode_'+end]/(tpot*value*1e9)
                if bounds['distinct_experts_per_layer_min']==bounds['distinct_experts_per_layer_max']:
                    payload=dict(gemm_weight_bytes=bounds['logical_bytes_per_decode_min']-bounds['kv_read_bytes'],
                        kv_read_bytes=bounds['kv_read_bytes'],average_kv_rows=bounds['average_kv_rows'],
                        logical_bytes_per_decode=bounds['logical_bytes_per_decode_min'])
                    row.update(payload,mbu_measured=row['mbu_measured_min'],mbu_nominal=row['mbu_nominal_min'])
                    row['mbu_scope']='inferred logical BF16 payload; B1 distinct experts equal top-k by invariant; not physical traffic'
        else:
            tpot = float(cell['tpot_ms']) / 1000
            if not math.isfinite(tpot) or tpot <= 0:
                raise ValueError('invalid TPOT')
            row.update(payload, mbu_measured=payload['logical_bytes_per_decode'] / (tpot * measured * 1e9),
                       mbu_nominal=payload['logical_bytes_per_decode'] / (tpot * args.nominal_gbps * 1e9))
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    print(f'{len(rows)} complete cells; MoE B1 top-k payload and bounds; B>1 point estimates require routing')


if __name__ == '__main__':
    main()
