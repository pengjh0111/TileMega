"""Load checkpoint tensors and apply the graph-emitted packing recipes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open

from .plan import Buffer, PlanLibrary


class Checkpoint:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.files: dict[str, Path] = {}
        for path in sorted(self.model_dir.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as stream:
                for name in stream.keys():
                    if name in self.files:
                        raise ValueError(f"duplicate checkpoint tensor {name}")
                    self.files[name] = path
        if not self.files:
            raise FileNotFoundError(f"no safetensors in {self.model_dir}")

    def tensor(self, name: str) -> torch.Tensor:
        # The tied vocabulary may be recorded only under the embedding name.
        if name == "lm_head.weight" and name not in self.files:
            name = "model.embed_tokens.weight"
        path = self.files.get(name)
        if path is None:
            raise KeyError(f"checkpoint has no {name}")
        with safe_open(path, framework="pt", device="cpu") as stream:
            return stream.get_tensor(name)


def _packed_cpu(recipe: Mapping[str, object], checkpoint: Checkpoint) -> torch.Tensor:
    kind = recipe["kind"]
    if kind == "alias":
        return checkpoint.tensor(str(recipe["source"]))
    sources = [checkpoint.tensor(str(name)) for name in recipe["sources"]]
    if not all(item.ndim == 2 for item in sources):
        raise ValueError("packed GEMM sources must be matrices")
    if len({item.shape[1] for item in sources}) != 1:
        raise ValueError("packed GEMM sources disagree on K")
    if kind == "qkv_group_interleave":
        if len(sources) != 3:
            raise ValueError("QKV recipe requires three weights")
        hkv, qperkv, dim = (int(recipe[key]) for key in
                              ("hkv", "qperkv", "head_dim"))
        q, k, v = sources
        if q.shape[0] != hkv * qperkv * dim or any(
                item.shape[0] != hkv * dim for item in (k, v)):
            raise ValueError("QKV recipe and checkpoint shapes disagree")
        return torch.cat([
            part for group in range(hkv) for part in (
                q[group * qperkv * dim:(group + 1) * qperkv * dim],
                k[group * dim:(group + 1) * dim],
                v[group * dim:(group + 1) * dim])], dim=0)
    if kind == "gate_up_interleave":
        if len(sources) != 2 or sources[0].shape != sources[1].shape:
            raise ValueError("gate/up recipe requires equally shaped weights")
        width = int(recipe["u"])
        gate, up = sources
        if width <= 0 or gate.shape[0] % width:
            raise ValueError("gate/up interleave width does not divide rows")
        return torch.stack((gate.reshape(-1, width, gate.shape[1]),
                            up.reshape(-1, width, up.shape[1])), dim=1
                           ).reshape(-1, gate.shape[1])
    raise ValueError(f"unsupported graph packing recipe {kind}")


def weight_recipes(*plans: PlanLibrary) -> dict[str, Buffer]:
    result: dict[str, Buffer] = {}
    for plan in plans:
        for buffer in plan.buffers:
            if not buffer.pack_json:
                continue
            prior = result.setdefault(buffer.name, buffer)
            if (prior.pack_json, prior.dtype, prior.elements_constant) != (
                    buffer.pack_json, buffer.dtype, buffer.elements_constant):
                raise ValueError(f"plans disagree on weight {buffer.name}")
    return result


def load_weights(model_dir: str | Path, *plans: PlanLibrary,
                 device: str | torch.device = "cuda") -> dict[str, torch.Tensor]:
    checkpoint = Checkpoint(model_dir)
    result: dict[str, torch.Tensor] = {}
    # The tied table is transferred once even if it has two plan references.
    aliases: dict[str, torch.Tensor] = {}
    for name, buffer in weight_recipes(*plans).items():
        recipe = json.loads(buffer.pack_json)
        tensor = _packed_cpu(recipe, checkpoint)
        if tensor.numel() != buffer.elements_constant:
            raise ValueError(f"{name} has {tensor.numel()} elements; plan expects "
                             f"{buffer.elements_constant}")
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        if recipe["kind"] == "alias":
            source = str(recipe["source"])
            if source == "lm_head.weight" and source not in checkpoint.files:
                source = "model.embed_tokens.weight"
            if source in aliases:
                result[name] = aliases[source]
                continue
            aliases[source] = tensor.contiguous().to(device)
            result[name] = aliases[source]
        else:
            result[name] = tensor.contiguous().to(device)
    return result
