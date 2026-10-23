"""Independent eager FIA-v2 candidate; never used by native TileMega code."""
import torch
import torch_npu


def fia_v2_forward(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
    if module.training or dropout or query.dtype!=torch.bfloat16 or key.dtype!=query.dtype or value.dtype!=query.dtype:
        raise ValueError('FIA candidate requires eval BF16 inference without dropout')
    if query.ndim!=4 or key.ndim!=4 or key.shape!=value.shape or query.shape[0]!=key.shape[0] or query.shape[-1]!=key.shape[-1] or query.shape[1]%key.shape[1]:
        raise ValueError('Invalid BNSD/GQA inference shape')
    if query.shape[2] not in (1,64) or key.shape[2]!=1088 or query.shape[-1] not in (64,128):
        raise ValueError('FIA candidate is bounded to the canonical static-cache request')
    if attention_mask is None or attention_mask.ndim!=4 or attention_mask.shape[-2:]!=(query.shape[2],key.shape[2]) or not attention_mask.dtype.is_floating_point:
        raise ValueError('Require the fixed model causal additive mask')
    # Transformers creates only zero/min-finite masks in this fixed no-padding
    # harness. FIA uses True for blocked entries and consumes GQA KV directly.
    blocked=(attention_mask<0).contiguous()
    output,_=torch_npu.npu_fused_infer_attention_score_v2(
        query.contiguous(),key.contiguous(),value.contiguous(),atten_mask=blocked,
        num_query_heads=query.shape[1],num_key_value_heads=key.shape[1],
        softmax_scale=scaling if scaling is not None else query.shape[-1]**-.5,
        input_layout='BNSD',sparse_mode=0,inner_precise=0)
    return output.transpose(1,2).contiguous(),None


def install(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if not hasattr(torch_npu,'npu_fused_infer_attention_score_v2'):
        raise ValueError('Installed torch_npu has no FIA-v2; no implicit fallback')
    ALL_ATTENTION_FUNCTIONS.register('as2_fia_v2',fia_v2_forward)
    model.config._attn_implementation='as2_fia_v2'
