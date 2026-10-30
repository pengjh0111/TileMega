"""Opt-in BF16 grouped-mm provider for an independent MetaX compile baseline.

This is a separately labelled missing-kernel repair. HF retains its grouping,
gating, routing and reduction; only its unsupported matrix multiply is replaced.
No TileMega plans or serving kernels are imported.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_kernel(A,W,O,Offsets,N:tl.constexpr,K:tl.constexpr,E:tl.constexpr,
                    AS0:tl.constexpr,AS1:tl.constexpr,WS0:tl.constexpr,
                    WS1:tl.constexpr,WS2:tl.constexpr,Workers:tl.constexpr,
                    BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=32):
    tile=tl.program_id(0)
    group_first=0
    first_tile=0
    ntiles=tl.cdiv(N,BN)
    for expert in range(E):
        group_end=tl.load(Offsets+expert)
        rows=group_end-group_first
        last_tile=first_tile+tl.cdiv(rows,BM)*ntiles
        while tile<last_tile:
            local=tile-first_tile
            rm=(local//ntiles)*BM+tl.arange(0,BM)
            rn=(local%ntiles)*BN+tl.arange(0,BN)
            rk=tl.arange(0,BK)
            accum=tl.full((BM,BN),0,tl.float32)
            for chunk in range(tl.cdiv(K,BK)):
                kk=chunk*BK+rk
                a=tl.load(A+(group_first+rm[:,None])*AS0+kk[None,:]*AS1,
                          (rm[:,None]<rows)&(kk[None,:]<K),other=0)
                w=tl.load(W+expert*WS0+kk[:,None]*WS1+rn[None,:]*WS2,
                          (kk[:,None]<K)&(rn[None,:]<N),other=0)
                accum=tl.dot(a,w,accum)
            tl.store(O+(group_first+rm[:,None])*N+rn[None,:],accum,
                     (rm[:,None]<rows)&(rn[None,:]<N))
            tile+=Workers
        group_first=group_end
        first_tile=last_tile


@torch.library.custom_op('metax_compile_baseline::grouped_mm',mutates_args=())
def grouped_mm(input:torch.Tensor,weight:torch.Tensor,offs:torch.Tensor)->torch.Tensor:
    if input.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16:
        raise ValueError('MetaX grouped-mm provider requires BF16 inputs and weights')
    if input.ndim!=2 or weight.ndim!=3 or input.shape[1]!=weight.shape[1]:
        raise ValueError('MetaX grouped-mm input/weight shape differs')
    if offs.ndim!=1 or offs.numel()!=weight.shape[0] or offs.dtype!=torch.int32 or not offs.is_contiguous():
        raise ValueError('MetaX grouped-mm requires one contiguous int32 end offset per expert')
    if input.device.type!='cuda' or weight.device!=input.device or offs.device!=input.device:
        raise ValueError('MetaX grouped-mm tensors must share the visible CUDA device')
    output=torch.empty((input.shape[0],weight.shape[2]),device=input.device,dtype=input.dtype)
    workers=torch.cuda.get_device_properties(input.device).multi_processor_count
    _grouped_kernel[(workers,)](input,weight,output,offs,weight.shape[2],input.shape[1],weight.shape[0],
        *input.stride(),*weight.stride(),workers,num_warps=4,num_stages=2)
    return output


@grouped_mm.register_fake
def _fake(input,weight,offs):
    return input.new_empty((input.shape[0],weight.shape[2]))


def install():
    from transformers.integrations import moe
    moe._grouped_mm=grouped_mm
    return {'grouped_mm_provider':'metax_triton_persistent_bf16',
            'grouped_mm_repair':'explicit opt-in replacement of the missing grouped-mm kernel',
            'expert_arithmetic':'HF grouped_mm grouping/gating/routing/reduction unchanged'}
