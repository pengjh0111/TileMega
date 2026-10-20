#!/usr/bin/env python3
"""Check a private venv's existing torch_npu stack on one selected device."""
import json
import os
import torch
import torch_npu

if torch.npu.device_count() != 1:
    raise RuntimeError('Expected one visible NPU; use as1_run.py')
torch.npu.set_device(0)
torch.manual_seed(20261016)
x = torch.randn(128, dtype=torch.bfloat16)
y = (x.npu() + 1).cpu()
reference = x + 1
if not torch.equal(y, reference):
    raise RuntimeError('BF16 addition does not match independent CPU result')
print(json.dumps({'status': 'passed', 'evidence': 'verified',
                  'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                  'physical_device_id': int(os.environ['AS1_PHYSICAL_DEVICE_ID']),
                  'logical_device_id': torch.npu.current_device(),
                  'visible_count': torch.npu.device_count(), 'elements': x.numel(),
                  'max_abs_error': (y.float()-reference.float()).abs().max().item(),
                  'tilemega_pipeline': False}))
