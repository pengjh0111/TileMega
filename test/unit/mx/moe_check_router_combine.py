"""Fresh-process legacy bit comparisons; scheduling is external and guarded."""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,time
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--libraries',type=Path,required=True);a=p.parse_args();
if __import__('os').environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':p.error('run through scripts/mx_gpu_job.sh')
import torch as t
t.set_num_threads(1);R=Path(__file__).resolve().parents[3];B=a.libraries.resolve();a.out.mkdir(parents=True,exist_ok=False)
class Dispatch(C.Structure):
 _fields_=[(n,C.c_void_p)for n in ['logits','n_active','active','row_count','rows','weights','u_gate_up','u_down']]
class Combine(C.Structure):
 _fields_=[(n,C.c_void_p)for n in ['partials','weights','residual','output','tickets']]
class TopKLegacy(C.Structure):
 _fields_=[(n,C.c_void_p)for n in ['partial_logits','partial_indices','indices','weights']]+[(n,C.c_uint)for n in ['tokens','experts','parts']]+[(n,C.c_uint64)for n in ['row_stride','part_stride','rank_stride']]+[('weights_fp32',C.c_bool)]
class CombineLegacy(C.Structure):
 _fields_=[(n,C.c_void_p)for n in ['partials','weights','residual','output','row_stats']]+[(n,C.c_uint)for n in ['tokens','channels']]+[(n,C.c_uint64)for n in ['token_stride','rank_stride','row_stride','stats_stride']]+[('rms_squares',C.c_void_p),('weights_fp32',C.c_bool)]
files={n:B/f for n,f in [('router','dispatch.so'),('combine','combine.so'),('legacy','legacy.so')]}
libs={n:C.CDLL(str(f))for n,f in files.items()};router=libs['router'].mx5_moe_dispatch_probe;router.argtypes=[Dispatch,C.c_int,C.c_int,C.c_int,C.c_void_p,C.POINTER(C.c_int),C.c_int];router.restype=C.c_int
combine=libs['combine'].mx5_moe_combine_probe;combine.argtypes=[Combine,C.c_uint,C.c_int,C.c_int,C.c_int,C.c_int,C.c_void_p,C.POINTER(C.c_int),C.c_int];combine.restype=C.c_int
legacytop=libs['legacy'].mx5_legacy_topk;legacytop.argtypes=[TopKLegacy,C.c_void_p];legacytop.restype=C.c_int
legacycombine=libs['legacy'].mx5_legacy_combine;legacycombine.argtypes=[CombineLegacy,C.c_void_p];legacycombine.restype=C.c_int
assert libs['legacy'].mx5_legacy_topk_operands_size()==C.sizeof(TopKLegacy)
assert libs['legacy'].mx5_legacy_combine_operands_size()==C.sizeof(CombineLegacy)
stream=t.cuda.current_stream().cuda_stream;queries=[];rows=[];g=t.Generator().manual_seed(2030)
def ptr(x):return x.data_ptr()
def observe(fn,args,label):
 resources=(C.c_int*4)();rc=fn(*args,stream,resources,1);assert rc==0,(label,'query',rc)
 assert resources[1]==resources[2]==0 and resources[3]==1,(label,list(resources))
 queries.append(dict(label=label,returncode=rc,MTreg=resources[0],stack=resources[1],static_smem=resources[2],resident=resources[3],launches=0));return resources
for b in [1,2,4,8,16]:
 for fp32 in [0,1]:
  dtype=t.float32 if fp32 else t.bfloat16
  for threads in [512,576]:
   for pattern in ['random','ties','ordered']:
    cpu=t.randn((b,128),generator=g).bfloat16() if pattern=='random' else (t.zeros((b,128),dtype=t.bfloat16)if pattern=='ties'else t.arange(128).reshape(1,128).expand(b,-1).bfloat16())
    logits=cpu.cuda();plog=cpu.float().cuda();pidx=t.arange(128,dtype=t.int32,device='cuda').repeat(b,1);idx=t.empty((b,8),device='cuda',dtype=t.int32);legacyweight=t.empty((b,8),device='cuda',dtype=dtype)
    lp=TopKLegacy(ptr(plog),ptr(pidx),ptr(idx),ptr(legacyweight),b,128,16,128,8,1,bool(fp32));assert legacytop(lp,stream)==0
    active=t.full((128,),-1,device='cuda',dtype=t.int32);count=active.clone();dispatch_rows=t.full((128,b,2),-1,device='cuda',dtype=t.int32);weight=t.empty((b,8),device='cuda',dtype=dtype);na=t.zeros(1,device='cuda',dtype=t.int32);ug=na.clone();ud=na.clone()
    dp=Dispatch(*map(ptr,[logits,na,active,count,dispatch_rows,weight,ug,ud]));label=dict(batch=b,weights_fp32=bool(fp32),threads=threads,pattern=pattern)
    if pattern=='random':observe(router,[dp,b,fp32,threads],label)
    rc=router(dp,b,fp32,threads,stream,None,0);assert rc==0,(label,rc);t.cuda.synchronize()
    assert t.equal(weight,legacyweight),(label,'weights bit mismatch')
    expected=idx.cpu();experts=sorted(set(expected.flatten().tolist()));assert na.item()==len(experts)
    assert ug.item()==len(experts)*24 and ud.item()==len(experts)*64
    assert active[:len(experts)].cpu().tolist()==experts
    actual=dispatch_rows.cpu();counts=count.cpu();first=[x.clone()for x in [active,count,dispatch_rows,weight,na,ug,ud]]
    for ordinal,expert in enumerate(experts):
     pairs=[(token,rank)for token in range(b)for rank in range(8)if int(expected[token,rank])==expert]
     assert counts[ordinal].item()==len(pairs)and actual[ordinal,:len(pairs)].tolist()==[list(x)for x in pairs],(label,ordinal,pairs)
    assert router(dp,b,fp32,threads,stream,None,0)==0;t.cuda.synchronize()
    assert all(t.equal(x,y)for x,y in zip(first,[active,count,dispatch_rows,weight,na,ug,ud])),(label,'dispatch nondeterminism')
    rows.append(dict(kind='router',**label,legacy_weights_bitwise=True,top8_and_dispatch=True,repeat_bitwise=True))
for b in [1,16]:
 for tn in [16,128]:
  for fp32 in [0,1]:
   dtype=t.float32 if fp32 else t.bfloat16
   for threads in [512,576]:
    for active in ([8]if b==1 else[8,82,104,128]):
     partial=t.randn((b,8,2048),generator=g).bfloat16().cuda();weight=t.randn((b,8),generator=g).to(dtype).cuda();residual=t.randn((b,2048),generator=g).bfloat16().cuda();out=t.full((b,2048),-91,device='cuda',dtype=t.bfloat16);ref=t.empty_like(out);ticket=t.zeros(128,device='cuda',dtype=t.int32)
     lp=CombineLegacy(ptr(partial),ptr(weight),ptr(residual),ptr(ref),None,b,2048,8*2048,2048,2048,0,None,bool(fp32));assert legacycombine(lp,stream)==0
     cp=Combine(ptr(partial),ptr(weight),ptr(residual),ptr(out),ptr(ticket));label=dict(batch=b,tn=tn,weights_fp32=bool(fp32),threads=threads,active=active)
     if active==8:observe(combine,[cp,active,b,tn,fp32,threads],label)
     for repeat in range(2):
      assert combine(cp,active,b,tn,fp32,threads,stream,None,0)==0;t.cuda.synchronize()
      assert t.equal(out[:,:tn],ref[:,:tn]),(label,'combine bit mismatch',repeat)
      assert t.all(out[:,tn:]==-91)and t.all(ticket==0),(label,'output bounds or ticket reset')
     rows.append(dict(kind='combine',**label,legacy_output_bitwise=True,repeat_bitwise=True))
result=dict(epoch=time.time(),PID=__import__('os').getpid(),rows=rows,queries=queries,zero_skips=True,passed=True,binaries={n:dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())for n,f in files.items()},scope='Router/topk/dispatch and combine legacy bit comparisons;not grouped GEMM, native model, timing, or 104CTA/2000iter publication litmus')
(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('MOE_ROUTER_COMBINE_LEGACY_PASS',len(rows),'queries',len(queries),flush=True)
