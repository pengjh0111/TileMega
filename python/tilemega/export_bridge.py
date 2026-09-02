#!/usr/bin/env python3
"""Thin, version-locked torch.export -> stable JSON bridge.

This file deliberately performs no operator classification, stage formation,
granularity choice, or coupling construction.  Those decisions live in the C++
TorchExportImporter and produce the CG dialect directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


SCHEMA = "tilemega.exported_program.v1"


def _shape(value: Any) -> list[str] | list[Any] | None:
    if isinstance(value, torch.Tensor):
        return [str(dim) for dim in value.shape]
    if isinstance(value, (tuple, list)):
        return [_shape(item) for item in value]
    return None


def _dtype(value: Any) -> str | list[Any] | None:
    if isinstance(value, torch.Tensor):
        return str(value.dtype)
    if isinstance(value, (tuple, list)):
        return [_dtype(item) for item in value]
    return None


def _guards(program: torch.export.ExportedProgram) -> list[str]:
    """The only adapter touching a private torch API; fail loudly on drift."""
    major_minor = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if major_minor != (2, 13):
        raise RuntimeError(
            "TileMega export guard adapter supports torch 2.13.x; got "
            f"{torch.__version__}. Update _guards() as one atomic adapter."
        )
    if not hasattr(program, "_guards_code"):
        raise RuntimeError(
            "torch 2.13 ExportedProgram has no _guards_code; guard semantics "
            "cannot be imported safely"
        )
    return list(program._guards_code)


def serialize(program: torch.export.ExportedProgram) -> dict[str, Any]:
    nodes = []
    for index, node in enumerate(program.graph.nodes):
        value = node.meta.get("val")
        nodes.append(
            {
                "index": index,
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "inputs": [item.name for item in node.all_input_nodes],
                "shape": _shape(value),
                "dtype": _dtype(value),
                "nn_module_stack": [
                    {"path": str(path), "type": str(module_type)}
                    for path, module_type in node.meta.get("nn_module_stack", {}).values()
                ],
                "source_fn_stack": [str(item) for item in node.meta.get("source_fn_stack", [])],
            }
        )
    def argument_name(spec: Any) -> str:
        name = getattr(spec.arg, "name", None)
        if not isinstance(name, str):
            raise RuntimeError(
                "TileMega stable bridge currently requires named graph-signature "
                f"arguments; got {spec.arg!r}"
            )
        return name

    signature_inputs = [
        {
            "name": argument_name(spec),
            "kind": spec.kind.name,
            "target": spec.target,
            "persistent": spec.persistent,
        }
        for spec in program.graph_signature.input_specs
    ]
    signature_outputs = [
        {
            "name": argument_name(spec),
            "kind": spec.kind.name,
            "target": spec.target,
        }
        for spec in program.graph_signature.output_specs
    ]
    return {
        "schema": SCHEMA,
        "torch_version": torch.__version__,
        "nodes": nodes,
        "range_constraints": {
            str(symbol): str(constraint)
            for symbol, constraint in program.range_constraints.items()
        },
        "guards": _guards(program),
        # Signature roles are serialization facts, not TileMega semantic
        # decisions. Keeping them structured avoids parsing torch's diagnostic
        # __str__ representation in the C++ importer.
        "signature": {
            "inputs": signature_inputs,
            "outputs": signature_outputs,
        },
        "graph_signature": str(program.graph_signature),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="torch.export .pt2 archive")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    program = torch.export.load(args.input)
    document = serialize(program)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    call_functions = [node for node in document["nodes"] if node["op"] == "call_function"]
    dependencies = sum(
        1
        for node in call_functions
        for source in node["inputs"]
        if any(other["name"] == source for other in call_functions)
    )
    print(json.dumps({"tasks": len(call_functions), "couplings": dependencies,
                      "guards": len(document["guards"])}))


if __name__ == "__main__":
    main()
