#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
nvcc=${CUDACXX:-/usr/local/cuda/bin/nvcc}
ptxas=$(dirname "$nvcc")/ptxas
mkdir -p "$here/raw" "$here/out"
inc=(-I"$root/third_party/cutlass/include" -I"$root/third_party/cutlass/test")

nsec() { date +%s%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.6f",(b-a)/1e9}'; }
: >"$here/raw/full_compile_seconds.txt"
for run in $(seq 1 20); do
  a=$(nsec)
  "$nvcc" -std=c++17 -O2 -arch=sm_89 -cubin "${inc[@]}" \
    "$here/compile_unit.cu" -o "$here/out/full_${run}.cubin" 2>/dev/null
  b=$(nsec); echo "$(elapsed "$a" "$b")" >>"$here/raw/full_compile_seconds.txt"
done

a=$(nsec)
"$nvcc" -std=c++17 -O2 -arch=compute_89 -ptx "${inc[@]}" \
  "$here/compile_unit.cu" -o "$here/out/unit.ptx" 2>/dev/null
b=$(nsec); echo "$(elapsed "$a" "$b")" >"$here/raw/frontend_ptx_seconds.txt"
a=$(nsec); "$ptxas" -arch=sm_89 "$here/out/unit.ptx" -o "$here/out/unit.cubin" 2>/dev/null
b=$(nsec); echo "$(elapsed "$a" "$b")" >"$here/raw/ptxas_seconds.txt"

a=$(nsec)
for i in 1 2 3 4; do
  "$nvcc" -std=c++17 -O2 -arch=sm_89 -cubin "${inc[@]}" \
    "$here/compile_unit.cu" -o "$here/out/seq_${i}.cubin" 2>/dev/null
done
b=$(nsec); echo "$(elapsed "$a" "$b")" >"$here/raw/sequential4_seconds.txt"
a=$(nsec)
pids=()
for i in 1 2 3 4; do
  "$nvcc" -std=c++17 -O2 -arch=sm_89 -cubin "${inc[@]}" \
    "$here/compile_unit.cu" -o "$here/out/par_${i}.cubin" 2>/dev/null & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
b=$(nsec); echo "$(elapsed "$a" "$b")" >"$here/raw/parallel4_seconds.txt"

sort -n "$here/raw/full_compile_seconds.txt" | \
  awk 'NR==10{a=$1} NR==11{printf "median_full_seconds=%.6f\n",(a+$1)/2}' \
  >"$here/raw/summary.txt"
median=$(sed -n 's/median_full_seconds=//p' "$here/raw/summary.txt")
{
  echo "frontend_ptx_seconds=$(cat "$here/raw/frontend_ptx_seconds.txt")"
  echo "ptxas_seconds=$(cat "$here/raw/ptxas_seconds.txt")"
  echo "sequential4_seconds=$(cat "$here/raw/sequential4_seconds.txt")"
  echo "parallel4_seconds=$(cat "$here/raw/parallel4_seconds.txt")"
  awk -v m="$median" 'BEGIN{printf "estimated_170_serial_seconds=%.3f\n",170*m}'
} >>"$here/raw/summary.txt"
cat "$here/raw/summary.txt"
