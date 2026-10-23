"""One observed eager GEMM trace; no device-wide profiling counters."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from torch.utils._python_dispatch import TorchDispatchMode
from tilemega.bench.as2_models import RUNS,controlled
from tilemega.bench.as2_native import digest
from tilemega.bench.as2_sdpa_dispatch import describe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--no-profile',action='store_true')
    p.add_argument('--compare-deterministic',action='store_true',help='Diagnostic only; never changes the model baseline')
    a=p.parse_args();a.case=controlled(a.case,[RUNS]);a.out=controlled(a.out,[RUNS])
    if a.out.exists():raise FileExistsError(a.out)
    case=json.loads((a.case/'case.json').read_text());m,n,k=case['shape_mnk']
    if any(digest(a.case/name)!=sha for name,sha in case['files'].items()):raise ValueError('Observed GEMM case changed')
    if m not in (1,64) or not ((n==2048 and k in (6144,8192)) or (n==3072 and k==8192)):
        raise ValueError('This diagnostic is bounded to the observed down-projection families')
    torch.set_num_threads(1);torch.npu.set_device(0)
    if torch.npu.device_count()!=1 or torch.npu.matmul.allow_hf32:raise ValueError('Require one NPU and full FP32 math')
    x=torch.from_file(str(a.case/'a.bin'),shared=False,size=m*k,dtype=torch.bfloat16).reshape(1,m,k).npu()
    w=torch.from_file(str(a.case/'b_nk.bin'),shared=False,size=n*k,dtype=torch.bfloat16).reshape(n,k).npu()
    a.out.mkdir();records=[]
    if a.compare_deterministic and not a.no_profile:raise ValueError('Use a separate unprofiled deterministic diagnostic')
    class Record(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            entry=dict(operator=str(func),args=describe(args),kwargs=describe(kwargs or {}))
            result=func(*args,**(kwargs or {}));entry['result']=describe(result);records.append(entry);return result
    config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,l2_cache=False,record_op_args=True)
    handler=torch_npu.profiler.tensorboard_trace_handler(str(a.out/'profiler'),analyse_flag=False)
    with torch.inference_mode():
        torch.nn.functional.linear(x,w);torch.npu.synchronize()
        if a.no_profile:
            with Record():result=torch.nn.functional.linear(x,w)
            torch.npu.synchronize()
        else:
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True,experimental_config=config,on_trace_ready=handler) as prof:
                with Record():result=torch.nn.functional.linear(x,w)
                torch.npu.synchronize()
            from torch_npu.profiler.analysis.prof_common_func._constant import Constant
            prof.prof_if.analyse(Constant.EXPORT_CHROME_TRACE,str(a.out/'trace.json'),max_process_number=1)
        raw=result.cpu().contiguous();(a.out/'output.bin').write_bytes(raw.view(torch.uint8).numpy().tobytes())
        deterministic=None
        if a.compare_deterministic:
            previous=torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True);other=torch.nn.functional.linear(x,w)
                torch.npu.synchronize();other=other.cpu().contiguous()
                (a.out/'deterministic_output.bin').write_bytes(other.view(torch.uint8).numpy().tobytes())
                native=torch.from_file(str(a.case/'model_native.bin'),shared=False,size=m*n,dtype=torch.bfloat16).reshape_as(other)
                deterministic=dict(different_from_default=int((other!=raw).sum()),different_from_native=int((other!=native).sum()),
                    scope='isolated diagnostic, not a baseline or performance choice')
            finally:torch.use_deterministic_algorithms(previous)
    record=dict(case_sha256=digest(a.case/'case.json'),scope='one warmed eager GEMM only; not model/performance admission',
        hardware_counters_enabled=False,analysis_workers=1,tilemega_pipeline=False,operators=records,
        matches_observed_reference=digest(a.out/'output.bin')==digest(a.case/'reference.bin'),
        torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,trace=str(a.out/'trace.json') if not a.no_profile else None,
        deterministic_comparison=deterministic)
    (a.out/'dispatch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
