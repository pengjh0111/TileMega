"""Grouped native kernel vs CPU FP32 GEMM and legacy combine, one process."""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,time,os,math
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--libraries',type=Path,required=True);a=p.parse_args();
if __import__('os').environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':p.error('run through scripts/mx_gpu_job.sh')
import torch as t
t.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=False);R=Path(__file__).resolve().parents[3];B=a.libraries.resolve();g=t.Generator().manual_seed(2030)
class Unit(C.Structure):_fields_=[(n,C.c_uint)for n in ['active','expert','tile','count','input_row']]
class Combine(C.Structure):_fields_=[(n,C.c_void_p)for n in ['partials','weights','residual','output','tickets']]
class Operands(C.Structure):
 _fields_=[(n,C.c_void_p)for n in ['a','w','out','active','row_count','rows']]+[('units',C.c_uint),('expert_stride',C.c_uint),('epsilon',C.c_float),('unit',Unit),('combine',Combine)]
class CombineLegacy(C.Structure):
 _fields_=[(n,C.c_void_p)for n in ['partials','weights','residual','output','row_stats']]+[(n,C.c_uint)for n in ['tokens','channels']]+[(n,C.c_uint64)for n in ['token_stride','rank_stride','row_stride','stats_stride']]+[('rms_squares',C.c_void_p),('weights_fp32',C.c_bool)]
so=B/'grouped.so';lib=C.CDLL(str(so));assert lib.mx5_moe_grouped_operands_size()==C.sizeof(Operands)
fn=lib.mx5_moe_grouped_probe;fn.argtypes=[Operands]+[C.c_int]*6+[C.c_void_p,C.POINTER(C.c_int),C.c_int];fn.restype=C.c_int
legacyso=B/'legacy.so';legacy=C.CDLL(str(legacyso));combine=legacy.mx5_legacy_combine;combine.argtypes=[CombineLegacy,C.c_void_p];combine.restype=C.c_int
stream=t.cuda.current_stream().cuda_stream
from tilemega.serving.weights import _packed_gpu
cpu_weights={};packed={}
for gu,n,k in [(1,1536,2048),(0,2048,768)]:
 cpu=(t.randn((128,n,k),generator=g)/math.sqrt(k)).bfloat16();cpu_weights[gu]=cpu
 recipe=dict(kind='tile_pages',layout='xcore_mma',tile_n=64,tile_k=32,source=dict(kind='alias',source='matrix'))
 packed[gu]=t.stack([_packed_gpu(recipe,lambda key,e=e:cpu[e])for e in range(128)]).cuda()
coords=[(1,1,1,512,1,0),(16,1,1,512,1,0),(1,2,2,576,1,0),(16,4,2,576,1,0),(16,8,1,576,1,0),(16,8,1,512,1,0),(1,1,1,512,0,0),(16,1,1,512,0,0),(1,2,2,576,0,2),(16,4,2,576,0,2),(16,8,1,576,0,4),(16,8,1,512,0,4)]
rows=[];queries=[]
for b,wn,depth,threads,gu,pw in coords:
 # Eight distinct experts per token, with overlaps and nonconsecutive expert IDs.
 route=t.stack([t.randperm(128,generator=g)[:8]for _ in range(b)]);experts=sorted(set(route.flatten().tolist()));active_count=len(experts);counts=[];mapping=t.zeros((128,b,2),dtype=t.int32)
 for ordinal,expert in enumerate(experts):
  pairs=[(token,rank)for token in range(b)for rank in range(8)if int(route[token,rank])==expert];counts.append(len(pairs));mapping[ordinal,:len(pairs)]=t.tensor(pairs,dtype=t.int32)
 active=t.tensor(experts,device='cuda',dtype=t.int32);count=t.tensor(counts,device='cuda',dtype=t.int32);mapping_gpu=mapping.cuda();n,k=(1536,2048)if gu else(2048,768);units=active_count*(n//64);assert wn<=(units+103)//104
 x=t.randn((b,k)if gu else(active_count,b,k),generator=g).bfloat16();x_gpu=x.cuda();out=t.full((active_count,b,768)if gu else(b,8,2048),-93,device='cuda',dtype=t.bfloat16)
 residual=t.randn((b,2048),generator=g).bfloat16().cuda();weight=t.rand((b,8),generator=g).float().cuda();merged=t.empty((b,2048),device='cuda',dtype=t.bfloat16);reference_merge=t.empty_like(merged);ticket=t.zeros(32,device='cuda',dtype=t.int32)
 cp=Combine(out.data_ptr(),weight.data_ptr(),residual.data_ptr(),merged.data_ptr(),ticket.data_ptr());ops=Operands(*[v.data_ptr()for v in [x_gpu,packed[gu],out,active,count,mapping_gpu]],units,n*k,1e-6,Unit(),cp);params=[b,wn,depth,threads,gu,pw]
 resource=(C.c_int*4)();rc=fn(ops,*params,stream,resource,1);assert rc==0,(params,'query',rc);assert resource[1]==resource[2]==0 and resource[3]==1
 queries.append(dict(coordinate=params,returncode=rc,MTreg=resource[0],stack=resource[1],static_smem=resource[2],resident=resource[3],launches=0))
 rc=fn(ops,*params,stream,None,0);assert rc==0,(params,rc);t.cuda.synchronize();actual=out.cpu();actual_merged=merged.clone()
 maximum=0.0
 for ordinal,expert in enumerate(experts):
  r=counts[ordinal];pairs=mapping[ordinal,:r].long();input=x[pairs[:,0]]if gu else x[ordinal,:r];ref=input.float()@cpu_weights[gu][expert].float().T
  if gu:
   ref*=t.rsqrt(input.float().square().mean(-1,keepdim=True)+1e-6);ref=ref.bfloat16().reshape(r,-1,2,16);gate,up=ref[:,:,0].float(),ref[:,:,1].float();expected=((gate/(1+(-gate).exp())).bfloat16().float()*up).bfloat16().reshape(r,768);value=actual[ordinal,:r]
  else:expected=ref.bfloat16();value=actual[pairs[:,0],pairs[:,1]]
  t.testing.assert_close(value.float(),expected.float(),rtol=.012,atol=.02);maximum=max(maximum,float((value.float()-expected.float()).abs().max()))
 if not gu:
  lp=CombineLegacy(out.data_ptr(),weight.data_ptr(),residual.data_ptr(),reference_merge.data_ptr(),None,b,2048,8*2048,2048,2048,0,None,True);assert combine(lp,stream)==0;t.cuda.synchronize();assert t.equal(merged,reference_merge),'native combine differs from actual legacy body';assert t.all(ticket==0)
 rc=fn(ops,*params,stream,None,0);assert rc==0;t.cuda.synchronize();assert t.equal(out.cpu(),actual),(params,'grouped nondeterminism')
 if not gu:assert t.equal(merged,actual_merged),(params,'combine nondeterminism')
 rows.append(dict(coordinate=params,n_active=active_count,FP32_reference_passed=True,max_abs_error=maximum,repeat_bitwise=True,legacy_combine_bitwise=not gu));print('GROUPED_CASE_PASS',params,active_count,maximum,flush=True)
result=dict(epoch=time.time(),PID=os.getpid(),passed=True,rows=rows,queries=queries,zero_skips=True,binary=dict(path=str(so),sha256=hashlib.sha256(so.read_bytes()).hexdigest()),legacy_binary=dict(path=str(legacyso),sha256=hashlib.sha256(legacyso.read_bytes()).hexdigest()),not_native_model_acceptance=True)
(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('MOE_GROUPED_PASS',len(rows),flush=True)
