#!/usr/bin/env python3
"""Summarize historical-KV effective bandwidth from the archived R11 trace.

The numerator counts every historical K/V element once per query block. It is
an effective byte rate, not a hardware DRAM counter: some bytes may hit L2.
"""

import csv
import io
import json
import statistics
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRACE = HERE / "page_vector_diagnostics"
CONFIGS = {
    "llama": HERE.parent / "MODELS/sources/llama_config_public_copy.json",
    "qwen3": HERE.parent / "MODELS/sources/qwen_config.json",
}


def read_member(archive, name):
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(name)
    return member.read().decode()


def summarize(archive, cell):
    model, batch_text = cell.split("_B")
    batch = int(batch_text)
    config = json.loads(CONFIGS[model].read_text())
    plan = json.loads((TRACE / "manifests" / f"{cell}.json").read_text())
    trace_header = json.loads(read_member(archive, f"{cell}/trace.log").splitlines()[0])
    assert trace_header["batch"] == batch and trace_header["past"] == 575
    assert plan["batch_lo"] == plan["batch_hi"] == batch
    assert plan["attention_query_rows"] == config["num_attention_heads"] // config["num_key_value_heads"]
    assert plan["seq"] == 1
    layers = config["num_hidden_layers"]
    heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    past = trace_header["past"]
    blocks = (plan["capacity"] + plan["attention_kv_block"] - 1) // plan["attention_kv_block"]
    stages_per_layer = 8 if blocks > 1 else 7
    stages = list(csv.DictReader(io.StringIO(read_member(
        archive, f"{cell}/chain/trace_v2.task_spaces.tsv")), delimiter="\t"))
    assert len(stages) == layers * stages_per_layer + 4
    spans_ns = []
    for layer in range(layers):
        stage = stages[3 + layer * stages_per_layer]
        assert int(stage["stage"]) == 3 + layer * stages_per_layer
        assert int(stage["tasks"]) == batch * heads * blocks
        spans_ns.append(int(stage["last_end_ns"]) - int(stage["first_start_ns"]))
    bytes_per_layer = batch * heads * past * head_dim * 4  # K and V, BF16
    aggregate_bytes = bytes_per_layer * layers
    aggregate_ns = sum(spans_ns)
    return {
        "cell": cell,
        "past": past,
        "layers": layers,
        "kv_blocks": blocks,
        "historical_kv_gb": round(aggregate_bytes / 1e9, 6),
        "attention_stage_sum_ms": round(aggregate_ns / 1e6, 6),
        "attention_stage_median_us": round(statistics.median(spans_ns) / 1e3, 3),
        "effective_historical_kv_gbps": round(aggregate_bytes / aggregate_ns, 3),
    }


def main():
    cells = ("llama_B1", "llama_B16", "qwen3_B1", "qwen3_B16")
    with tarfile.open(TRACE / "raw.tar.xz", "r:xz") as archive:
        rows = [summarize(archive, cell) for cell in cells]
    output = TRACE / "attention_bandwidth.tsv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
