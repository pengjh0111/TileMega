"""One BF16 matrix product verifies a private framework stack, not a model."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
import torch_npu


def main():
    out=Path(sys.argv[1]).absolute();root=Path('/home/d50064638/junhuip/TileMega/runs/as2')
    if out.resolve()!=out or out==root or not out.is_relative_to(root):raise ValueError('Use a canonical AS2 artifact path')
    if os.environ.get('AS2_PHYSICAL_DEVICE_ID') not in ('0','1'):raise ValueError('Use the no-lock single-card wrapper')
    if torch.version.cuda is not None or torch.npu.device_count()!=1:raise ValueError('Require CPU-build PyTorch and exactly one visible NPU')
    torch.set_num_threads(1);torch.npu.set_device(0)
    sku=torch.npu.get_device_name(0)
    if not sku.startswith('Ascend950PR'):raise ValueError('Unexpected target SKU: '+sku)
    generator=torch.Generator().manual_seed(20261016)
    x=(torch.randn(16,256,generator=generator)*.125).bfloat16()
    w=(torch.randn(128,256,generator=generator)*.02).bfloat16()
    reference=(x.float()@w.float().T).bfloat16()
    result=torch.nn.functional.linear(x.npu(),w.npu());torch.npu.synchronize();result=result.cpu()
    error=(result.float()-reference.float()).abs();failed=(error>.001+.0078125*reference.float().abs())|~torch.isfinite(result)
    out.mkdir(exist_ok=False)
    receipt=dict(schema='tilemega.as2.framework_stack_probe.v1',status='fail' if failed.any() else 'pass',
        scope='one BF16 GEMM stack probe; no engine/model/performance admission',
        shape_mnk=[16,128,256],sku=sku,physical_device_id=int(os.environ['AS2_PHYSICAL_DEVICE_ID']),logical_device_id=0,
        torch=torch.__version__,torch_npu=torch_npu.__version__,torch_file=torch.__file__,torch_npu_file=torch_npu.__file__,
        python=sys.executable,prefix=sys.prefix,max_abs=float(error.max()),failed_elements=int(failed.sum()),
        output_sha256=hashlib.sha256(result.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),device_locked=False)
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    return int(bool(failed.any()))


if __name__=='__main__':raise SystemExit(main())
