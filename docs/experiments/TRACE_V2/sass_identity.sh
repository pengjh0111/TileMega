#!/usr/bin/env bash
# EX-D1 H2: with TILEMEGA_TRACE_V2 undefined, a default build must emit the
# same SASS it emitted before trace v2 existed.  The comparison is against the
# baseline commit's headers, compiled here with identical flags, so a diff can
# only come from this round's edits.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
out="${here}/sass_identity"
base_commit="${BASE_COMMIT:-4cdebdf13495bb9a11c2be839755d4faff0bea1f}"
base_tree="${BASE_TREE:-/tmp/tilemega-sass-base}"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
cuobjdump="${CUOBJDUMP:-/usr/local/cuda-12.8/bin/cuobjdump}"

rm -rf "${base_tree}" "${out}"
mkdir -p "${base_tree}" "${out}/bin"
git -C "${repo}" archive "${base_commit}" | tar -x -C "${base_tree}"

# The generated model sources and CUTLASS come from the working tree in both
# arms; only the include path for tilemega's own headers differs.
common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)

status=0
for model in gqa2 mha4; do
  src="${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu"
  for arm in base head; do
    inc="${repo}/include"
    [[ "${arm}" == base ]] && inc="${base_tree}/include"
    "${nvcc}" "${common[@]}" -I"${inc}" "${src}" "${link[@]}" \
      -o "${out}/bin/${model}_${arm}"
    "${cuobjdump}" --dump-sass "${out}/bin/${model}_${arm}" \
      > "${out}/${model}_${arm}.sass"
    for kernel in tilemega_l1_kernel tilemega_l2_kernel; do
      grep -q "${kernel}" "${out}/${model}_${arm}.sass" || {
        echo "SASS_IDENTITY model=${model} arm=${arm} missing=${kernel}" >&2
        exit 2
      }
    done
  done
  if diff -u "${out}/${model}_base.sass" "${out}/${model}_head.sass" \
      > "${out}/${model}.diff"; then
    echo "SASS_IDENTITY model=${model} identical=1 bytes=$(wc -c < "${out}/${model}_head.sass")"
  else
    echo "SASS_IDENTITY model=${model} identical=0 diff_lines=$(wc -l < "${out}/${model}.diff")"
    status=1
  fi
done

printf 'base_commit\t%s\nhead_commit\t%s\n' "${base_commit}" \
  "$(git -C "${repo}" rev-parse HEAD)" > "${out}/meta.tsv"
# The dumps are ~6.8 MB each and are not committed; their digests and the
# empty diffs are the evidence, and the script regenerates the dumps.
(cd "${out}" && sha256sum ./*.sass) > "${out}/sha256.tsv"
exit "${status}"
