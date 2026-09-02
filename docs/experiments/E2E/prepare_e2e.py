#!/usr/bin/env python3
"""Create deterministic L0 fixtures from V-H's real ExportedProgram model."""

from __future__ import annotations

import argparse
import array
import json
import sys
from pathlib import Path

import torch


def write_f32(path: Path, tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().contiguous().float()
    with path.open("wb") as stream:
        array.array("f", value.reshape(-1).tolist()).tofile(stream)
    return {"file": path.name, "shape": list(value.shape), "elements": value.numel()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vh-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
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
    hidden = torch.randn(1, 4, 512)
    caches = [torch.randn(1, 2, 3, 128) for _ in range(4)]
    inputs = [hidden, *caches]
    with torch.no_grad():
        outputs = model(*inputs)

    files: dict[str, dict] = {}
    files["input_hidden"] = write_f32(args.out / "input_hidden.bin", hidden)
    for index, cache in enumerate(caches):
        files[f"input_cache_{index}"] = write_f32(
            args.out / f"input_cache_{index}.bin", cache
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
        files[f"input_{name}"] = write_f32(
            args.out / f"input_{name}.bin", value
        )
    for name, value in program.state_dict.items():
        key = "state_" + name.replace(".", "_")
        files[key] = write_f32(args.out / f"{key}.bin", value)
    output_names = ["hidden", "k0", "v0", "k1", "v1"]
    for name, value in zip(output_names, outputs):
        files[f"reference_{name}"] = write_f32(
            args.out / f"reference_{name}.bin", value
        )
    for index, value in enumerate(outputs):
        files[f"reference_{index}"] = write_f32(
            args.out / f"reference_{index}.bin", value
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
            "seq": 4,
            "past": 3,
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
