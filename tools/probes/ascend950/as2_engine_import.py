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
        # This is the exact KV-write API that failed before private NNAL setup.
        torch.ops.load_library(str(Path(torch_npu.__file__).parent/'lib/libop_plugin_atb.so'))
        generator=torch.Generator().manual_seed(20261016)
        key=torch.randn((2,8,128),generator=generator).bfloat16().npu()
        value=torch.randn((2,8,128),generator=generator).bfloat16().npu()
        key_cache=torch.zeros((2,128,8,128),dtype=torch.bfloat16,device='npu:0')
        value_cache=torch.zeros_like(key_cache)
        slots=torch.tensor([0,129],dtype=torch.int32,device='npu:0')
        torch_npu._npu_reshape_and_cache(key=key,value=value,key_cache=key_cache,
                                      value_cache=value_cache,slot_indices=slots)
        torch.npu.synchronize()
        if not torch.equal(key_cache.view(256,8,128)[slots.long()].cpu(),key.cpu()) or not torch.equal(value_cache.view(256,8,128)[slots.long()].cpu(),value.cpu()):
            raise ValueError('ATB KV-write payload mismatch')
        identity=dict(version=metadata.version('sglang'),module=sglang.__file__,engine=Engine.__module__,
                      custom_ops=custom_ops.__file__,kernel=sgl_kernel_npu.__file__,
                      atb_home=os.environ.get('ATB_HOME_PATH'),atb_cache_probe='two rows at slots0/129 bitwise')
    torch.npu.synchronize()
    out.mkdir(exist_ok=False)
    receipt=dict(schema='tilemega.as2.engine_import.v1',status='pass',engine=args.engine,
        identity=identity,python=sys.executable,torch=torch.__version__,torch_npu=torch_npu.__version__,
        sku=torch.npu.get_device_name(0),physical_device_id=int(os.environ['AS2_PHYSICAL_DEVICE_ID']),
        logical_device_id=0,device_locked=False,model_execution_verified=False,performance_admitted=False)
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
