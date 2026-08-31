#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
nvcc=${CUDACXX:-/usr/local/cuda/bin/nvcc}
mkdir -p "$here/raw"
"$nvcc" -std=c++17 -O2 -arch=sm_89 -cubin -Xptxas=-v \
  "$here/smem_union.cu" -o "$here/raw/smem_union.cubin" \
  2>"$here/raw/ptxas.txt"
python3 "$root/scripts/ptxas_parse.py" "$here/raw/ptxas.txt" \
  >"$here/raw/resources.jsonl"
: >"$here/raw/occupancy.txt"
for kernel in dispatch_2 dispatch_3 dispatch_5 dispatch_8 nested_dispatch \
              loop_carried_union loop_carried_separate; do
  "$root/tools/tilemega-occupancy" --cubin "$here/raw/smem_union.cubin" \
    --kernel "$kernel" | sed "s/^/kernel=$kernel /" >>"$here/raw/occupancy.txt"
done
cat "$here/raw/resources.jsonl"
cat "$here/raw/occupancy.txt"
