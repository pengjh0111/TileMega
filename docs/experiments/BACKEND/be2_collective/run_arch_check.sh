#!/usr/bin/env bash
# R8 BE-2 / A-f: compile the GEMM collective for every supported architecture
# and run each check on the CPU. No GPU of the target architecture is needed;
# the point is that the types instantiate and report what they selected.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
OUT="${OUT:-$REPO/docs/experiments/BACKEND/be2_collective}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
: >"$OUT/arch_check.tsv"
printf 'arch\tnvcc_arch\tcompiled\toutput\n' >>"$OUT/arch_check.tsv"
for pair in "800 sm_80" "890 sm_89" "900 sm_90" "1000 sm_100" "1200 sm_120"; do
  set -- $pair
  id=$1; arch=$2
  if "$NVCC" -std=c++17 -O2 -arch="$arch" -DTILEMEGA_CHECK_ARCH="$id" \
      -DTILEMEGA_MODEL_BF16=1 \
      -I"$REPO/include" -I"$REPO/third_party/cutlass/include" \
      -I"$REPO/third_party/cutlass/tools/util/include" \
      -I"$REPO/third_party/cutlass/test" \
      "$OUT/arch_check.cu" -o "/tmp/tilemega_arch_check_$id" \
      >"$OUT/build_$arch.log" 2>&1; then
    line="$(/tmp/tilemega_arch_check_$id)"
    printf '%s\t%s\t1\t%s\n' "$arch" "$arch" "$line" >>"$OUT/arch_check.tsv"
    echo "$line"
  else
    reason="$(grep -m1 'error' "$OUT/build_$arch.log" | cut -c1-160)"
    printf '%s\t%s\t0\t%s\n' "$arch" "$arch" "$reason" >>"$OUT/arch_check.tsv"
    echo "ARCH_COLLECTIVE arch=$arch COMPILE_FAILED $reason"
  fi
done
