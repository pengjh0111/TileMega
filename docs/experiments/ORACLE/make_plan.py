#!/usr/bin/env python3
"""Write one exact generator-owned runtime plan for an ORACLE point."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--tile-m", type=int, required=True)
p.add_argument("--tile-n", type=int, required=True)
p.add_argument("--tile-k", type=int, required=True)
p.add_argument("--stages", type=int, required=True)
p.add_argument("--split-k", type=int, required=True)
a = p.parse_args()
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({
    "schema": "tilemega.runtime_variants.v1",
    "variants": [{
        "seq_begin": 1, "seq_end": 2048,
        "rope_tile_per_block": True,
        "kv_tile_per_block": True,
        "activation_tile_per_block": True,
        "combiner_tile_per_block": True,
        "uniform": {
            "tile_m": a.tile_m, "tile_n": a.tile_n,
            "tile_k": a.tile_k, "stages": a.stages,
            "split_k": a.split_k,
        },
    }],
}, indent=2) + "\n", encoding="utf-8")
