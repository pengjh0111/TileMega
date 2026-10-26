"""Verify private engine imports on the wrapper-selected card; no model claim."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine',choices=('vllm','sglang'))
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    root=Path('/home/d50064638/junhuip/TileMega/runs/as2')
    out=args.out.absolute()
    if out.resolve()!=out or out==root or not out.is_relative_to(root):
        raise ValueError('Use a canonical AS2 artifact path')
    if os.environ.get('AS2_PHYSICAL_DEVICE_ID') not in ('0','1'):
        raise ValueError('Use the no-lock single-card wrapper')
    import torch
    import torch_npu
    if torch.version.cuda is not None or torch.npu.device_count()!=1:
        raise ValueError('Require CPU-build PyTorch and one visible NPU')
    torch.npu.set_device(0)
    if args.engine=='vllm':
        import vllm
        from vllm.platforms import current_platform
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        if current_platform.device_type!='npu':
            raise ValueError('Ascend plugin was not selected')
        from vllm_ascend._build_info import __device_type__
        if __device_type__!='A5':raise ValueError('Require the official 950 plugin')
        identity=dict(version=vllm.__version__,module=vllm.__file__,
                      platform=type(current_platform).__name__,wheel_device_type=__device_type__)
    else:
        import sglang
        import custom_ops
        import sgl_kernel_npu
        from sglang.srt.entrypoints.engine import Engine
        identity=dict(version=metadata.version('sglang'),module=sglang.__file__,engine=Engine.__module__,
                      custom_ops=custom_ops.__file__,kernel=sgl_kernel_npu.__file__)
    torch.npu.synchronize()
    out.mkdir(exist_ok=False)
    receipt=dict(schema='tilemega.as2.engine_import.v1',status='pass',engine=args.engine,
        identity=identity,python=sys.executable,torch=torch.__version__,torch_npu=torch_npu.__version__,
        sku=torch.npu.get_device_name(0),physical_device_id=int(os.environ['AS2_PHYSICAL_DEVICE_ID']),
        logical_device_id=0,device_locked=False,model_execution_verified=False,performance_admitted=False)
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
