#!/usr/bin/env python3
"""Materialize a seq/past fixture from the exported four-layer MHA model."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_exporter(path: Path):
    spec = importlib.util.spec_from_file_location("tilemega_mha_export", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--past", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    exporter = load_exporter(
        args.repo / "docs/experiments/P3_GENERALIZATION/export_second.py")
    program = torch.export.load(args.program)
    dtype = next(iter(program.state_dict.values())).dtype
    torch.manual_seed(20260902)
    hidden = torch.randn(1, args.seq, 512, dtype=dtype)
    caches = [torch.randn(1, 4, args.past, 128, dtype=dtype) for _ in range(8)]
    inputs = (hidden, *caches)
    with torch.no_grad():
        outputs = program.module()(*inputs)
    for spec, value in zip(
        (item for item in program.graph_signature.input_specs
         if item.kind.name == "USER_INPUT"), inputs
    ):
        exporter.write_tensor(args.out / f"input_{spec.arg.name}.bin", value,
                              dtype)
    for name, value in program.state_dict.items():
        exporter.write_tensor(
            args.out / ("state_" + name.replace(".", "_") + ".bin"), value,
            dtype)
    for index, value in enumerate(outputs):
        exporter.write_tensor(args.out / f"reference_{index}.bin", value,
                              dtype)
    (args.out / "manifest.json").write_text(json.dumps({
        "seq": args.seq, "past": args.past, "layers": 4, "heads": 4,
        "kv_heads": 4, "dtype": str(dtype), "structure": "MHA without GQA",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
