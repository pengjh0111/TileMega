#!/usr/bin/env python3
"""Recompute G-8 directly from the four raw CG tensor-footprint dumps."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCES = ROOT / "docs/experiments/MODELS/sources"
CASES = (
    ("llama", 1, 64, "llama_config_public_copy.json"),
    ("llama", 16, 1086, "llama_config_public_copy.json"),
    ("qwen3", 1, 64, "qwen_config.json"),
    ("qwen3", 16, 1086, "qwen_config.json"),
)


def config_weight_bytes(config: dict, qk_norm: bool) -> int:
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    head_dim = config.get("head_dim", hidden // config["num_attention_heads"])
    q_width = config["num_attention_heads"] * head_dim
    kv_width = config["num_key_value_heads"] * head_dim
    projection = hidden * (q_width + 2 * kv_width + hidden +
                           3 * intermediate)
    normalizations = 2 * hidden + (2 * head_dim if qk_norm else 0)
    embedding = config["vocab_size"] * hidden
    untied_head = 0 if config["tie_word_embeddings"] else embedding
    return 2 * (embedding + untied_head + layers *
                (projection + normalizations) + hidden)


def main() -> int:
    passed = True
    print("model\tbatch\tpast\tCG_weight_bytes\tconfig_weight_bytes\tweight_error"
          "\tCG_KV_bytes\tconfig_KV_bytes\tKV_error\tT_dram_ns\tT_compute_ns"
          "\tstatus\tevidence")
    for model, batch, past, config_file in CASES:
        directory = HERE / f"{model}_B{batch}_p{past}"
        config = json.loads((SOURCES / config_file).read_text())
        with (directory / "floor_tensors.tsv").open() as stream:
            tensors = list(csv.DictReader(stream, delimiter="\t"))
        with (directory / "floor.tsv").open() as stream:
            timing = next(csv.DictReader(stream, delimiter="\t"))
        weight = sum(int(row["no_producer_read_bytes"]) for row in tensors
                     if row["tensor"].endswith(".weight"))
        kv = sum(int(row["no_producer_read_bytes"]) for row in tensors
                 if row["tensor"].startswith("kv_cache."))
        expected_weight = config_weight_bytes(config, model == "qwen3")
        head_dim = config.get("head_dim", config["hidden_size"] //
                              config["num_attention_heads"])
        expected_kv = (batch * past * config["num_hidden_layers"] * 2 *
                       config["num_key_value_heads"] * head_dim * 2)
        weight_error = abs(weight / expected_weight - 1)
        kv_error = abs(kv / expected_kv - 1)
        good = weight_error <= 0.005 and kv_error <= 0.005
        passed &= good
        print(f"{model}\t{batch}\t{past}\t{weight}\t{expected_weight}\t"
              f"{weight_error:.9g}\t{kv}\t{expected_kv}\t{kv_error:.9g}\t"
              f"{timing['dram_ns']}\t{timing['compute_ns']}\t"
              f"{'PASS' if good else 'FAIL'}\t{directory.relative_to(ROOT)}")
    print(f"G-8 {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
