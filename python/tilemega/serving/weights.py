"""Load checkpoint tensors and apply the graph-emitted packing recipes."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open

from .plan import Buffer, PlanLibrary
from tilemega.dnn.weights import DNN_RECIPES, pack as pack_dnn, recipe_sources as dnn_sources
from tilemega.moe.weights import is_expert_recipe, pack_experts, recipe_sources as expert_sources


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
        self.hashes: dict[str, str] = {}

    def tensor(self, name: str) -> torch.Tensor:
        # The tied vocabulary may be recorded only under the embedding name.
        if name == "lm_head.weight" and name not in self.files:
            name = "model.embed_tokens.weight"
        path = self.files.get(name)
        if path is None:
            raise KeyError(f"checkpoint has no {name}")
        with safe_open(path, framework="pt", device="cpu") as stream:
            return stream.get_tensor(name)

    def tensor_sha(self, name: str) -> str:
        if name not in self.hashes:
            tensor = self.tensor(name).contiguous().view(torch.uint8).numpy()
            self.hashes[name] = hashlib.sha256(memoryview(tensor)).hexdigest()
        return self.hashes[name]


def _conv_krsc(weight: torch.Tensor, padded_channels: int) -> torch.Tensor:
    if weight.ndim != 4 or any(extent <= 0 for extent in weight.shape):
        raise ValueError('conv_krsc requires a nonempty OIHW tensor')
    outputs, channels, rows, columns = weight.shape
    if ((channels <= 4 and padded_channels not in (4, 8)) or
            (channels > 4 and padded_channels != (channels + 7) // 8 * 8)):
        raise ValueError('conv_krsc channel padding differs from the input layout')
    packed = weight.new_zeros((outputs, rows, columns, padded_channels))
    packed[..., :channels] = weight.permute(0, 2, 3, 1)
    return packed


def _packed_cpu(recipe: Mapping[str, object], checkpoint: Checkpoint) -> torch.Tensor:
    kind = recipe["kind"]
    if is_expert_recipe(recipe):
        return _packed_gpu(recipe, checkpoint.tensor)
    if kind in ('tile_pages', 'fold_rmsnorm'):
        return _packed_gpu(recipe, checkpoint.tensor)
    if kind in DNN_RECIPES:
        return pack_dnn(recipe, checkpoint.tensor, lambda value: _packed_gpu(value, checkpoint.tensor))
    if kind == "alias":
        return checkpoint.tensor(str(recipe["source"]))
    if kind == 'conv_krsc':
        return _conv_krsc(checkpoint.tensor(str(recipe['source'])),
                          int(recipe['padded_channels']))
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


def _recipe_sources(recipe: Mapping[str, object]) -> tuple[str, ...]:
    kind = recipe['kind']
    if is_expert_recipe(recipe):
        return expert_sources(recipe)
    if kind in DNN_RECIPES:
        return dnn_sources(recipe, _recipe_sources)
    if kind == 'tile_pages' or kind == 'fold_rmsnorm':
        nested = _recipe_sources(recipe['source'])
        return nested + ((str(recipe['norm']),) if kind == 'fold_rmsnorm' else ())
    if kind in ('alias', 'conv_krsc'):
        return (str(recipe['source']),)
    return tuple(str(name) for name in recipe['sources'])


def _packed_gpu(recipe: Mapping[str, object], source) -> torch.Tensor:
    kind = recipe['kind']
    if is_expert_recipe(recipe):
        return pack_experts(recipe,source,lambda matrix,tn,tk:_packed_gpu(
            dict(kind='tile_pages',tile_n=tn,tile_k=tk,source=dict(kind='alias',source='matrix')),
            lambda name:matrix))
    if kind in DNN_RECIPES:
        return pack_dnn(recipe, source, lambda value: _packed_gpu(value, source))
    if kind == 'alias':
        return source(str(recipe['source']))
    if kind == 'conv_krsc':
        return _conv_krsc(source(str(recipe['source'])), int(recipe['padded_channels']))
    if kind == 'fold_rmsnorm':
        weight = _packed_gpu(recipe['source'], source)
        norm = source(str(recipe['norm']))
        if norm.ndim != 1 or norm.numel() != weight.shape[1]:
            raise ValueError('fold_rmsnorm width mismatch')
        # Bound the transient FP32 buffer while row-major and tile-page BF16
        # versions of the same weight coexist during plan creation.
        out = torch.empty_like(weight)
        norm_f32 = norm.float()[None, :]
        for first in range(0, weight.shape[0], 256):
            last = min(first + 256, weight.shape[0])
            out[first:last] = (weight[first:last].float() * norm_f32).to(torch.bfloat16)
        return out
    if kind == 'tile_pages':
        weight = _packed_gpu(recipe['source'], source)
        tn, tk = int(recipe['tile_n']), int(recipe['tile_k'])
        if (weight.ndim not in (2, 4) or tn <= 0 or tk <= 0 or tn % 8 or
                (tk not in (16, 32) and tk % 64)):
            raise ValueError('invalid tile_pages source or geometry')
        if weight.ndim == 4:
            n, r, s, cp = weight.shape
            if (not all((n, r, s, cp)) or tk not in (16, 32, 64, 128) or
                    cp % 4 or (cp < tk and tk % cp)):
                raise ValueError('invalid convolution page geometry')
            # Each filter position has its own channel tail. Flattening KRSC
            # first would merge that tail into the next position's K tile.
            if cp >= tk:
                issued = weight.new_zeros((n, r, s, (cp + tk - 1) // tk * tk))
                issued[..., :cp] = weight
                weight = issued
            weight = weight.reshape(n, -1)
        n, k = weight.shape
        nt, kt = (n + tn - 1) // tn, (k + tk - 1) // tk
        padded = torch.zeros((nt * tn, kt * tk), device=weight.device,
                             dtype=torch.bfloat16)
        padded[:n, :k] = weight
        logical = padded.reshape(nt, tn, kt, tk).permute(0, 2, 1, 3)
        index = torch.arange(tn * tk, device=weight.device)
        row, col = index // tk, index % tk
        if tk < 64:
            # Swizzle<B,3,3> takes its XOR source from linear address bit 6.
            # At TK=16/32 that is row bit 2/1, rather than the lowest row bit.
            atom = (row % 8) * tk + col
            perm = (8 * tk * (row // 8) +
                    (atom ^ (((atom >> 6) & (tk // 8 - 1)) << 3)))
        else:
            perm = (512 * (row // 8 + (tn // 8) * (col // 64)) +
                    64 * (row % 8) + 8 * ((col % 64 // 8) ^ (row % 8)) + col % 8)
        packed = torch.empty((nt, kt, tn * tk), device=weight.device,
                             dtype=torch.bfloat16)
        packed[..., perm] = logical.reshape(nt, kt, tn * tk)
        return packed.reshape(-1)
    weights = [source(str(name)) for name in recipe['sources']]
    if len({item.shape[1] for item in weights}) != 1:
        raise ValueError('packed sources disagree on K')
    if kind == 'qkv_group_interleave':
        q, k, v = weights
        hkv, qperkv, dim = (int(recipe[key]) for key in
                              ('hkv', 'qperkv', 'head_dim'))
        if q.shape[0] != hkv * qperkv * dim or any(
                item.shape[0] != hkv * dim for item in (k, v)):
            raise ValueError('QKV recipe and checkpoint shapes disagree')
        return torch.cat([part for group in range(hkv) for part in (
            q[group * qperkv * dim:(group + 1) * qperkv * dim],
            k[group * dim:(group + 1) * dim],
            v[group * dim:(group + 1) * dim])], dim=0)
    if kind == 'gate_up_interleave':
        gate, up = weights
        width = int(recipe['u'])
        if gate.shape != up.shape or width <= 0 or gate.shape[0] % width:
            raise ValueError('gate/up interleave shape mismatch')
        return torch.stack((gate.reshape(-1, width, gate.shape[1]),
                            up.reshape(-1, width, up.shape[1])), dim=1
                           ).reshape(-1, gate.shape[1])
    raise ValueError(f'unsupported graph packing recipe {kind}')


def load_weights(model_dir: str | Path, *plans: PlanLibrary,
                 device: str | torch.device = "cuda") -> dict[str, torch.Tensor]:
    checkpoint = Checkpoint(model_dir)
    result: dict[str, torch.Tensor] = {}
    sources: dict[tuple[str, bool], torch.Tensor] = {}
    packed: dict[tuple[tuple[str, ...], str], torch.Tensor] = {}
    def source(name: str, preserve: bool = False, cache: bool = True) -> torch.Tensor:
        key = (name, preserve)
        if key not in sources or not cache:
            original = checkpoint.tensor(name)
            tensor = original.to(device=device,
                dtype=original.dtype if preserve else torch.bfloat16).contiguous()
            if not cache:
                return tensor
            sources[key] = tensor
        return sources[key]
    def dnn(recipe):
        return recipe['kind'] in DNN_RECIPES or (isinstance(recipe.get('source'), dict) and dnn(recipe['source']))
    for name, buffer in weight_recipes(*plans).items():
        recipe = json.loads(buffer.pack_json)
        identity = tuple(checkpoint.tensor_sha(item) for item in _recipe_sources(recipe))
        key = (identity, json.dumps(recipe, sort_keys=True))
        if key not in packed:
            packed[key] = _packed_gpu(recipe, lambda name: source(name, dnn(recipe),
                cache=not is_expert_recipe(recipe))).contiguous()
        # Aliases intentionally retain their source through the result tensor.
        # Other recipes release source storage before packing the next weight.
        sources.clear()
        tensor = packed[key]
        if tensor.numel() != buffer.elements_constant:
            raise ValueError(f"{name} has {tensor.numel()} elements; plan expects "
                             f"{buffer.elements_constant}")
        dtype = {0: torch.bfloat16, 1: torch.float32, 2: torch.int32, 3: torch.int64}.get(buffer.dtype)
        if dtype is None:
            raise ValueError(f'unsupported weight buffer dtype {buffer.dtype}')
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        result[name] = tensor
    return result
