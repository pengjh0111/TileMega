#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=${TILEMEGA_SOURCE_DIR:-$(cd "$here/../../.." && pwd)}
nvcc=${CUDACXX:-$(command -v nvcc || true)}
if [[ -z "$nvcc" ]]; then
  for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    if [[ -x "$candidate" ]]; then nvcc=$candidate; break; fi
  done
fi
[[ -x "$nvcc" ]] || { echo "nvcc not found" >&2; exit 2; }

out="$here/raw"
mkdir -p "$out"
: > "$out/matrix.tsv"
printf 'arch\tstages\ttask_smem_bytes\tcollective\tmain_status\tadvanced_status\n' >> "$out/matrix.tsv"

for arch in 80 89 90 120; do
  config="$root/configs/targets/sm_${arch}.json"
  stages=$(python3 - "$config" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
r = c["resources"]
print(min(8, max(0, (r["max_dynamic_smem_per_cta"] - 4096) // 16384)))
PY
)
  smem=$((stages * 16384))
  case "$arch" in
    80|89) collective="cp.async_multistage" ;;
    90) collective="TMA_warp-specialized" ;;
    120) collective="TMA_warp-specialized_SM120_MMA_no_tcgen05" ;;
  esac

  log="$out/sm_${arch}.ptxas.txt"
  if "$nvcc" -std=c++17 -arch="sm_${arch}" -cubin -lineinfo \
      -Xptxas=-v -DTILEMEGA_STAGES="$stages" \
      -I"$root/include" -I"$root/third_party/cutlass/include" \
      "$here/crosscompile.cu" -o "$out/sm_${arch}.cubin" 2>"$log"; then
    main_status=PASS
  else
    main_status=FAIL
  fi

  advanced_status=not_applicable
  if [[ "$arch" == 90 || "$arch" == 120 ]]; then
    advanced_log="$out/sm_${arch}_cluster.ptxas.txt"
    if "$nvcc" -std=c++17 -arch="sm_${arch}" -cubin -lineinfo -Xptxas=-v \
        -I"$root/include" -I"$root/third_party/cutlass/include" \
        "$here/cluster_path.cu" -o "$out/sm_${arch}_cluster.cubin" \
        2>"$advanced_log"; then
      advanced_status=PASS
    else
      advanced_status=FAIL
    fi
  fi
  printf 'sm_%s\t%s\t%s\t%s\t%s\t%s\n' "$arch" "$stages" "$smem" \
    "$collective" "$main_status" "$advanced_status" >> "$out/matrix.tsv"
done

cat "$out/matrix.tsv"
if grep -q $'\tFAIL' "$out/matrix.tsv"; then exit 1; fi
