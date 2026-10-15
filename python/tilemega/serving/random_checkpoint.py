"""Supervise deterministic random HF checkpoints held only in Linux memfds.

Run this file directly to keep baseline environments independent of TileMega.
These finite random weights support execution checks, not pretrained C-1 claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

ALGORITHM = "tilemega-r16-name-seeded-uniform-v1"
DTYPES = {"torch.bfloat16": ("BF16", 2), "torch.float32": ("F32", 4),
          "torch.float16": ("F16", 2), "torch.int64": ("I64", 8),
          "torch.int32": ("I32", 4), "torch.bool": ("BOOL", 1)}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensor_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{ALGORITHM}:{seed}:{name}".encode()).digest()[:8], "little") & ((1 << 63)-1)


def describe(config_directory):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config_directory = Path(config_directory)
    config = AutoConfig.from_pretrained(str(config_directory), local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.tie_weights()
    canonical, aliases = {}, {}
    for name, tensor in model.named_parameters(remove_duplicate=False):
        previous = canonical.setdefault(id(tensor), name)
        if previous != name:
            aliases[name] = previous
    tensors = []
    for name, tensor in model.state_dict().items():
        if name in aliases:
            continue
        dtype = str(tensor.dtype)
        if dtype not in DTYPES:
            raise ValueError(f"unsupported checkpoint dtype: {name} {dtype}")
        tensors.append({"name": name, "shape": list(tensor.shape), "dtype": DTYPES[dtype][0],
                        "torch_dtype": dtype, "bytes": tensor.numel()*DTYPES[dtype][1]})
    if not tensors:
        raise ValueError("model has no persistent tensors")
    return {"algorithm": ALGORITHM, "config_sha256": digest((config_directory/"config.json").read_bytes()),
            "architecture": type(model).__name__, "total_bytes": sum(t["bytes"] for t in tensors),
            "tensor_count": len(tensors), "aliases": aliases, "tensors": tensors,
            "torch_version": torch.__version__}


def partition(tensors, shard_bytes):
    if shard_bytes < 1:
        raise ValueError("shard_bytes must be positive")
    result, pending, size = [], [], 0
    for tensor in tensors:
        if pending and size + tensor["bytes"] > shard_bytes:
            result.append(pending); pending, size = [], 0
        pending.append(tensor); size += tensor["bytes"]
    if pending:
        result.append(pending)
    return result


def tensor_chunks(tensor, seed, chunk_bytes):
    import torch
    dtype = getattr(torch, tensor["torch_dtype"].split(".")[-1])
    elements = math.prod(tensor["shape"])
    # A fixed FP32 RNG quantum makes bytes independent of I/O chunk sizing.
    quantum = 65536
    generator = torch.Generator(device="cpu").manual_seed(tensor_seed(seed, tensor["name"]))
    norm = tensor["name"].endswith("weight") and "norm" in tensor["name"].lower()
    bias = tensor["name"].endswith("bias")
    for first in range(0, elements, quantum):
        count = min(quantum, elements-first)
        if norm:
            value = torch.ones(count, dtype=dtype)
        elif bias or not dtype.is_floating_point:
            value = torch.zeros(count, dtype=dtype)
        else:
            value = torch.empty(count, dtype=torch.float32).uniform_(-0.02, 0.02, generator=generator).to(dtype)
        data = memoryview(value.view(torch.uint8).numpy())
        for offset in range(0, len(data), chunk_bytes):
            yield data[offset:offset+chunk_bytes]


def _write(fd, data):
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short memfd write")
        view = view[written:]


class RandomCheckpoint:
    def __init__(self, source, destination, *, seed=0, shard_bytes=512 << 20,
                 chunk_bytes=1 << 20, max_bytes=None):
        self.source, self.directory = Path(source), Path(destination)
        self.seed, self.shard_bytes, self.chunk_bytes = seed, shard_bytes, chunk_bytes
        self.max_bytes = max_bytes
        self.fds, self.links = [], []
        self.manifest = None

    def __enter__(self):
        if not hasattr(os, "memfd_create"):
            raise RuntimeError("random checkpoints require Linux memfd_create")
        if self.chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        self.manifest = describe(self.source)
        if self.max_bytes is not None and self.manifest["total_bytes"] > self.max_bytes:
            raise ValueError("checkpoint exceeds declared max_bytes")
        shards = partition(self.manifest["tensors"], self.shard_bytes)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest.update(seed=self.seed, supervisor_pid=os.getpid(), state="building",
                             weight_semantics="shared synthetic random checkpoint; pretrained C-1 unavailable",
                             source=str(self.source.resolve()), shards=[])
        try:
            for filename in ("config.json", "generation_config.json"):
                if (self.source/filename).is_file():
                    shutil.copyfile(self.source/filename, self.directory/filename)
            mapping = {}
            for number, tensors in enumerate(shards, 1):
                filename = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
                fd = os.memfd_create(filename, flags=os.MFD_CLOEXEC)
                self.fds.append(fd)
                header, offset = {"__metadata__": {"format": "pt"}}, 0
                for tensor in tensors:
                    header[tensor["name"]] = {"dtype": tensor["dtype"], "shape": tensor["shape"],
                                               "data_offsets": [offset, offset+tensor["bytes"]]}
                    offset += tensor["bytes"]
                    mapping[tensor["name"]] = filename
                encoded = json.dumps(header, separators=(",", ":")).encode()
                encoded += b" "*((-len(encoded)) % 8)
                prefix = struct.pack("<Q", len(encoded)) + encoded
                _write(fd, prefix)
                shard_hash = hashlib.sha256(prefix)
                for tensor in tensors:
                    tensor_hash = hashlib.sha256()
                    for chunk in tensor_chunks(tensor, self.seed, self.chunk_bytes):
                        _write(fd, chunk)
                        tensor_hash.update(chunk); shard_hash.update(chunk)
                    tensor["sha256"] = tensor_hash.hexdigest()
                link = self.directory/filename
                target = f"/proc/{os.getpid()}/fd/{fd}"
                link.symlink_to(target)
                self.links.append((link, target))
                self.manifest["shards"].append({"file": filename, "bytes": len(prefix)+offset,
                    "data_bytes": offset, "sha256": shard_hash.hexdigest(),
                    "exceeds_requested_shard_bytes": offset > self.shard_bytes})
            index = {"metadata": {"total_size": self.manifest["total_bytes"]}, "weight_map": mapping}
            (self.directory/"model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
            identity = {key: self.manifest[key] for key in ("algorithm", "config_sha256", "seed", "aliases", "tensors")}
            self.manifest["checkpoint_sha256"] = digest(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
            self.manifest["state"] = "live"
            self._save()
            return self
        except BaseException:
            self.manifest["state"] = "failed"
            self.close()
            raise

    def _save(self):
        (self.directory/"random_checkpoint.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def close(self):
        for path, target in self.links:
            if path.is_symlink() and os.readlink(path) == target:
                path.unlink()
        self.links.clear()
        for fd in self.fds:
            os.close(fd)
        self.fds.clear()
        if self.manifest is not None and self.directory.exists():
            if self.manifest["state"] != "failed":
                self.manifest["state"] = "closed"
            self._save()

    def __exit__(self, *error):
        self.close()


def validate_live_checkpoint(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"random_checkpoint.json").read_text())
    if manifest["state"] != "live":
        raise RuntimeError("random checkpoint supervisor is closed or failed; weights are unavailable")
    for shard in manifest["shards"]:
        if not (directory/shard["file"]).is_file():
            raise RuntimeError("random checkpoint memfd is unavailable; supervisor may have exited")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-mib", type=int, default=512)
    parser.add_argument("--chunk-mib", type=int, default=1)
    parser.add_argument("--max-gib", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe(args.config), indent=2))
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.out is None or not command:
        parser.error("materialization requires --out and -- command; use --describe for shapes only")
    if args.max_gib is not None and (not math.isfinite(args.max_gib) or args.max_gib <= 0):
        parser.error("max-gib must be positive and finite")
    with RandomCheckpoint(args.config, args.out, seed=args.seed,
            shard_bytes=args.shard_mib << 20, chunk_bytes=args.chunk_mib << 20,
            max_bytes=int(args.max_gib*(1 << 30)) if args.max_gib is not None else None) as checkpoint:
        env = dict(os.environ, TILEMEGA_RANDOM_CHECKPOINT=str(checkpoint.directory.resolve()))
        print(json.dumps({key: checkpoint.manifest[key] for key in
            ("state", "checkpoint_sha256", "tensor_count", "total_bytes", "supervisor_pid")}), flush=True)
        # Children open /proc/<supervisor>/fd paths, so descriptor inheritance is
        # unnecessary and all checkpoint allocations remain owned by this scope.
        result = subprocess.run(command, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
