#!/usr/bin/env python3
"""Create deterministic L0 fixtures from V-H's real ExportedProgram model."""

from __future__ import annotations

import argparse
import array
import json
import sys
from pathlib import Path

import torch


def write_tensor(path: Path, tensor: torch.Tensor, dtype: torch.dtype) -> dict:
    value = tensor.detach().cpu().contiguous().to(dtype)
    with path.open("wb") as stream:
        if dtype == torch.bfloat16:
            # Python's array has no BF16 type; preserve PyTorch's raw 16-bit
            # representation without routing through NumPy.
            words = [item & 0xFFFF for item in value.view(torch.int16).reshape(-1).tolist()]
            array.array("H", words).tofile(stream)
        else:
            array.array("f", value.float().reshape(-1).tolist()).tofile(stream)
    return {"file": path.name, "shape": list(value.shape),
            "elements": value.numel(), "dtype": str(dtype)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vh-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seq", type=int, default=4)
    parser.add_argument("--past", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = json.loads((args.vh_raw / "report.json").read_text())
    cg = json.loads((args.vh_raw / "cg.json").read_text())
    whitelist = {name for name, _ in report["operators"]}
    outside = sorted(
        {task["operator"] for task in cg["task_spaces"] if task["operator"] not in whitelist}
    )
    if outside:
        raise RuntimeError(f"operators outside V-H whitelist: {outside}")

    torch.manual_seed(20260901)
    program = torch.export.load(args.vh_raw / "exported_program.pt2")
    # ExportedProgram.module() is already an inference GraphModule. Its eval()
    # method intentionally raises in current torch.export releases.
    model = program.module()
    dtype = next(iter(program.state_dict.values())).dtype
    hidden = torch.randn(1, args.seq, 512, dtype=dtype)
    caches = [torch.randn(1, 2, args.past, 128, dtype=dtype) for _ in range(4)]
    inputs = [hidden, *caches]
    with torch.no_grad():
        outputs = model(*inputs)

    files: dict[str, dict] = {}
    files["input_hidden"] = write_tensor(args.out / "input_hidden.bin", hidden, dtype)
    for index, cache in enumerate(caches):
        files[f"input_cache_{index}"] = write_tensor(
            args.out / f"input_cache_{index}.bin", cache, dtype
        )
    # Stable bridge names: the C++ importer can derive fixture paths from the
    # structured ExportGraphSignature without knowing that this model calls
    # its first input "hidden" or numbers KV tensors in pairs. Keep the legacy
    # aliases above for the handwritten E2E reference.
    for spec, value in zip(
        (item for item in program.graph_signature.input_specs
         if item.kind.name == "USER_INPUT"),
        inputs,
    ):
        name = spec.arg.name
        files[f"input_{name}"] = write_tensor(
            args.out / f"input_{name}.bin", value, dtype
        )
    for name, value in program.state_dict.items():
        key = "state_" + name.replace(".", "_")
        files[key] = write_tensor(args.out / f"{key}.bin", value, dtype)
    output_names = ["hidden", "k0", "v0", "k1", "v1"]
    for name, value in zip(output_names, outputs):
        files[f"reference_{name}"] = write_tensor(
            args.out / f"reference_{name}.bin", value, dtype
        )
    for index, value in enumerate(outputs):
        files[f"reference_{index}"] = write_tensor(
            args.out / f"reference_{index}.bin", value, dtype
        )

    e2e_graph = {
        "source": "V-H ExportedProgram",
        "theta": cg["theta"],
        "range_constraints": cg["range_constraints"],
        "task_spaces": cg["task_spaces"],
        "couplings": cg["couplings"],
        "fixed_g": {
            "token_tile": 1,
            "hidden_tile": 128,
            "head_tile": 1,
            "head_dim_tile": 128,
        },
        "unsupported_operators": outside,
    }
    (args.out / "e2e_graph.json").write_text(json.dumps(e2e_graph, indent=2))
    manifest = {
        "seed": 20260901,
        "config": {
            "layers": 2,
            "batch": 1,
            "seq": args.seq,
            "past": args.past,
            "dtype": str(dtype),
            "hidden": 512,
            "intermediate": 1024,
            "heads": 4,
            "kv_heads": 2,
            "head_dim": 128,
        },
        "frontend": {
            "source_task_spaces": len(cg["task_spaces"]),
            "source_couplings": len(cg["couplings"]),
            "whitelist_size": len(whitelist),
            "unsupported": outside,
        },
        "files": files,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["frontend"], indent=2))


if __name__ == "__main__":
    main()
