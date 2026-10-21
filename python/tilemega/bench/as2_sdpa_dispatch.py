"""Record one real eager SDPA dispatch without device-wide profiling counters."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torch_npu
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.cache_utils import StaticCache
from tilemega.bench.as2_eager import load_model, digest
from tilemega.bench.as2_models import RUNS, controlled, DENSE


def describe(value):
    if isinstance(value,torch.Tensor):
        return dict(shape=list(value.shape),dtype=str(value.dtype),device=str(value.device),stride=list(value.stride()))
    if isinstance(value,(list,tuple)):return [describe(v) for v in value]
    if isinstance(value,dict):return {str(k):describe(v) for k,v in value.items()}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return str(value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=DENSE,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',action='store_true',help='One operator only, no AIC metrics, one analysis worker')
    a=p.parse_args();out=controlled(a.out,[RUNS]);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.npu.set_device(0)
    model,ids,manifest,weight_hash=load_model(a.model,'sdpa','fused')
    cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=1088,device=torch.device('npu:0'),dtype=torch.bfloat16)
    position=torch.arange(64,device='npu:0',dtype=torch.int64);tokens=ids[:1].to('npu:0')
    original=F.scaled_dot_product_attention;records=[];captured={}
    class Record(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {};entry=dict(operator=str(func),args=describe(args),kwargs=describe(kwargs))
            result=func(*args,**kwargs);entry['result']=describe(result);records.append(entry);return result
    def call(*args,**kwargs):
        if captured:return original(*args,**kwargs)
        captured.update(query=args[0],key=args[1],value=args[2],kwargs=kwargs)
        if a.profile:
            torch.npu.synchronize()
            config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,l2_cache=False,record_op_args=True)
            handler=torch_npu.profiler.tensorboard_trace_handler(str(out/'profiler'),analyse_flag=False)
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,
                    experimental_config=config,on_trace_ready=handler) as prof:
                with Record():result=original(*args,**kwargs)
                torch.npu.synchronize()
            from torch_npu.profiler.analysis.prof_common_func._constant import Constant
            prof.prof_if.analyse(Constant.EXPORT_CHROME_TRACE,str(out/'trace.json'),max_process_number=1)
        else:
            with Record():result=original(*args,**kwargs)
        captured['output']=result
        return result
    F.scaled_dot_product_attention=call
    try:
        with torch.inference_mode():
            result=model(input_ids=tokens,past_key_values=cache,use_cache=True,cache_position=position,
                position_ids=position[None,:],logits_to_keep=1,return_dict=True)
        torch.npu.synchronize()
    finally:F.scaled_dot_product_attention=original
    if not captured:raise ValueError('No SDPA call was observed')
    files=[]
    for name in ('query','key','value','output'):
        tensor=captured[name].detach().cpu().contiguous();file=out/(name+'.bin')
        file.write_bytes(tensor.view(torch.uint8).numpy().tobytes())
        files.append(dict(name=name,file=file.name,**describe(tensor),sha256=digest(file)))
    mask=captured['kwargs'].get('attn_mask')
    if mask is not None:
        tensor=mask.detach().cpu().contiguous();file=out/'mask.bin';file.write_bytes(tensor.view(torch.uint8).numpy().tobytes())
        files.append(dict(name='mask',file=file.name,**describe(tensor),sha256=digest(file)))
    receipt=dict(schema='tilemega.as2.sdpa_dispatch.v1',status='recorded_diagnostic_not_timing',model_id=a.model,
        config_sha256=manifest['config_sha256'],weight_manifest_sha256=weight_hash,input_sha256=manifest['input_ids']['sha256'],
        scope='first attention of full-layer-count prefill64, static KV1088, SDPA/fused norm',
        sdpa_kwargs=describe(captured['kwargs']),operators=records,files=files,
        torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,device_locked=False)
    receipt.update(profile_requested=a.profile,profile_scope='one first-layer attention call only',
        hardware_metrics_enabled=False,analysis_workers=1,device_trace=str(out/'trace.json') if (out/'trace.json').is_file() else None)
    (out/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],operators=[x['operator'] for x in records])))


if __name__=='__main__':main()
