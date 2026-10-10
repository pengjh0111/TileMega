"""Choose cross-phase layout constraints without loading checkpoint tensors."""
from pathlib import Path
import json
from tilemega.cache import file_sha


def automatic_layout_policy(model, target, workload):
    config=json.loads((Path(model)/'config.json').read_text())
    capacity=target.get('resources',{}).get('dram_capacity_bytes',0)
    result=dict(evidence='inferred',require_shared_layout=False,
                device_capacity_bytes=capacity,
                scope='BF16 checkpoint plus a second expert layout and request-state lower bound; excludes workspaces')
    if config.get('model_type')!='qwen3_moe' or not capacity:
        return result
    if type(capacity) is not int or capacity<0:
        raise ValueError('invalid device DRAM capacity')
    index=Path(model)/'model.safetensors.index.json'
    if not index.is_file():
        result['reason']='checkpoint index has no allocation byte count'
        return result
    weight_bytes=json.loads(index.read_text()).get('metadata',{}).get('total_size')
    if type(weight_bytes) is not int or weight_bytes<=0:
        raise ValueError('checkpoint index has no positive integer total_size')
    layers=config['num_hidden_layers']; hidden=config['hidden_size']
    experts=config.get('num_experts',config.get('num_local_experts'))
    if type(experts) is not int or experts<=0 or \
            config.get('num_local_experts',experts)!=experts:
        raise ValueError('checkpoint expert-count aliases disagree or are invalid')
    expert_bytes=layers*experts*3*hidden*config['moe_intermediate_size']*2
    if expert_bytes>weight_bytes:
        raise ValueError('expert geometry exceeds checkpoint allocation')
    batch=max(workload['batch']); tokens=workload['prompt_len']+workload['max_new_tokens']
    dim=config.get('head_dim',hidden//config['num_attention_heads'])
    state=layers*2*batch*config['num_key_value_heads']*tokens*dim*2 + \
        batch*tokens*4+2*tokens*dim*2
    lower=weight_bytes+expert_bytes+state
    result.update(checkpoint_bytes=weight_bytes,second_expert_layout_bytes=expert_bytes,
                  request_state_bytes=state,two_layout_lower_bound_bytes=lower,
                  config_sha256=file_sha(Path(model)/'config.json'),index_sha256=file_sha(index),
                  require_shared_layout=lower>capacity)
    return result
