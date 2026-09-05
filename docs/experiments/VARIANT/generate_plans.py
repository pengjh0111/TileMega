#!/usr/bin/env python3
"""Generate the 1/2/4/8/16 variant resource-curve plans."""
import json
import sys
from pathlib import Path


SHAPES = [
    (32, 16, 16, 2), (32, 32, 16, 2), (32, 64, 16, 2),
    (64, 64, 16, 2), (32, 128, 16, 2), (64, 128, 16, 2),
    (128, 128, 16, 2), (64, 128, 16, 3), (128, 128, 16, 3),
    (128, 256, 16, 2), (256, 256, 16, 2), (128, 256, 16, 3),
    (256, 256, 16, 3), (256, 256, 32, 2), (128, 256, 32, 3),
    (256, 256, 32, 3),
]
COUNTS = (1, 2, 4, 8, 16)


def ownership():
    return {
        "rope_tile_per_block": True,
        "kv_tile_per_block": True,
        "activation_tile_per_block": True,
        "combiner_tile_per_block": True,
    }


def gemm(shape):
    m, n, k, stages = shape
    return {"tile_m": m, "tile_n": n, "tile_k": k,
            "stages": stages, "split_k": 1}


def write(path, variants):
    path.write_text(json.dumps({
        "schema": "tilemega.runtime_variants.v1", "variants": variants,
    }, indent=2) + "\n", encoding="utf-8")


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for count in COUNTS:
        # Same operator family changes uniformly at runtime intervals.
        variants = []
        for index, shape in enumerate(SHAPES[:count]):
            begin = 1 + index * 2048 // count
            end = (index + 1) * 2048 // count
            variants.append({"seq_begin": begin, "seq_end": end,
                             **ownership(), "uniform": gemm(shape)})
        write(out / f"same_{count}.json", variants)

        # Different operators use the shapes concurrently. More than fourteen
        # unique shapes needs a second interval because the fixture has 14 GEMMs.
        variants = []
        remaining = list(SHAPES[:count])
        begin = 1
        while remaining:
            selected, remaining = remaining[:14], remaining[14:]
            end = 1024 if remaining else 2048
            table = [gemm(selected[i % len(selected)]) for i in range(14)]
            variants.append({"seq_begin": begin, "seq_end": end,
                             **ownership(), "gemms": table})
            begin = end + 1
        write(out / f"multi_{count}.json", variants)


if __name__ == "__main__":
    main()
