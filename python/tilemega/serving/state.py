"""Device-resident request state shared by prefill and decode plans."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig

from .plan import PlanLibrary


@dataclass
class ServingState:
    tensors: dict[str, torch.Tensor]
    kv_storage: torch.Tensor
    tokens: torch.Tensor

    @property
    def pointers(self) -> dict[str, int]:
        return {name: tensor.data_ptr() for name, tensor in self.tensors.items()}


def rotary_tables(config: object, capacity: int,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    architecture = getattr(config, "architectures", [""])[0]
    if architecture == "LlamaForCausalLM":
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        module = LlamaRotaryEmbedding(config, device=device)
    elif architecture == "Qwen3ForCausalLM":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        module = Qwen3RotaryEmbedding(config, device=device)
    else:
        raise ValueError(f"unsupported rotary position module {architecture}")
    head_dim = int(config.head_dim)
    dummy = torch.empty((1, 1, 1, head_dim), dtype=torch.bfloat16,
                        device=device)
    position_ids = torch.arange(capacity, dtype=torch.long,
                                device=device)[None, :]
    with torch.no_grad():
        cos, sin = module(dummy, position_ids)
    if cos.shape != (1, capacity, head_dim):
        raise ValueError(f"unexpected rotary table shape {cos.shape}")
    return cos[0].contiguous(), sin[0].contiguous()


def allocate_state(model_dir: str | Path, batch: int,
                   prefill: PlanLibrary, decode: PlanLibrary,
                   device: str | torch.device = "cuda") -> ServingState:
    if prefill.info.phase != 0 or decode.info.phase != 1:
        raise ValueError("plans must be ordered prefill, decode")
    if prefill.info.capacity != decode.info.capacity:
        raise ValueError("prefill and decode capacities differ")
    capacity = prefill.info.capacity
    if not prefill.info.batch_lo <= batch <= prefill.info.batch_hi or not (
            decode.info.batch_lo <= batch <= decode.info.batch_hi):
        raise ValueError("batch outside one plan's solved interval")
    device = torch.device(device)
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    layers = int(config.num_hidden_layers)
    hkv = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    kv_storage = torch.empty((layers, 2, batch, hkv, capacity, head_dim),
                             dtype=torch.bfloat16, device=device)
    tokens = torch.empty((batch, capacity), dtype=torch.int32, device=device)
    cos, sin = rotary_tables(config, capacity, device)
    result: dict[str, torch.Tensor] = {
        "serving.tokens": tokens,
        "serving.rope_cos": cos,
        "serving.rope_sin": sin,
    }
    for layer in range(layers):
        result[f"kv_cache.k.{layer}"] = kv_storage[layer, 0]
        result[f"kv_cache.v.{layer}"] = kv_storage[layer, 1]
    for plan in (prefill, decode):
        for buffer in plan.buffers:
            if buffer.role != 1 or buffer.pack_json:
                continue
            tensor = result.get(buffer.name)
            if tensor is None:
                raise KeyError(f"unknown external request buffer {buffer.name}")
            expected_dtype = {0: torch.bfloat16, 1: torch.float32,
                              2: torch.int32}[buffer.dtype]
            if tensor.dtype != expected_dtype or tensor.numel() != (
                    buffer.elements_constant + batch * buffer.elements_per_batch):
                raise ValueError(f"{buffer.name} conflicts with plan metadata")
            if not tensor.is_contiguous():
                raise ValueError(f"{buffer.name} is not contiguous")
    return ServingState(result, kv_storage, tokens)
