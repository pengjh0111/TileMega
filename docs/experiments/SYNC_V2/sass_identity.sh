#!/usr/bin/env bash
# R3 H2: with TILEMEGA_WAIT_POLICY undefined, a default build must emit the
# same SASS the round-two baseline emitted.  The comparison is against the
# baseline commit's headers, compiled here with identical flags, so a diff can
# only come from this round's edits.
#
# The wait policy is written as compile-time constants rather than as a second
# code path, so the off state is not textually guarded at the two ClusterSync
# sites: it relies on `WaitBackoff::Pause()` folding to the `__nanosleep(64)`
# it replaces.  That fold is the thing this script checks -- a non-empty diff
# means the fold did not happen, not that the policy is wrong.
#
# The `on` arm is a control in the other direction: it must *differ*, or the
# switch is reaching nothing and every later measurement of it would be noise.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
out="${here}/sass_identity"
base_commit="${BASE_COMMIT:-4b37f940a4f6900c936f321514e3459e731f77ec}"
base_tree="${BASE_TREE:-/tmp/tilemega-sync-v2-base}"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
cuobjdump="${CUOBJDUMP:-/usr/local/cuda-12.8/bin/cuobjdump}"

rm -rf "${base_tree}" "${out}"
mkdir -p "${base_tree}" "${out}/bin"
git -C "${repo}" archive "${base_commit}" | tar -x -C "${base_tree}"

# H5: the on arm's constants come from the target, not from this script.
policy="$("${build}/tools/tilemega-wait-policy" sm_89 bf16 "${repo}")"
echo "SASS_IDENTITY policy_flags=${policy}"

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)

status=0
for model in gqa2 mha4; do
  src="${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu"
  for arm in base head on; do
    inc="${repo}/include"
    [[ "${arm}" == base ]] && inc="${base_tree}/include"
    flags=()
    [[ "${arm}" == on ]] && read -r -a flags <<< "${policy}"
    "${nvcc}" "${common[@]}" -I"${inc}" "${flags[@]+"${flags[@]}"}" \
      "${src}" "${link[@]}" -o "${out}/bin/${model}_${arm}"
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

  if diff -u "${out}/${model}_head.sass" "${out}/${model}_on.sass" \
      > "${out}/${model}_on.diff"; then
    echo "SASS_SWITCH model=${model} reached=0 -- the policy changed no code" >&2
    status=1
  else
    echo "SASS_SWITCH model=${model} reached=1 diff_lines=$(wc -l < "${out}/${model}_on.diff")"
  fi
done

printf 'base_commit\t%s\nhead_commit\t%s\npolicy_flags\t%s\n' "${base_commit}" \
  "$(git -C "${repo}" rev-parse HEAD)" "${policy}" > "${out}/meta.tsv"
# The dumps are ~6.8 MB each and are not committed; their digests, the empty
# H2 diffs and the non-empty switch diffs are the evidence.
(cd "${out}" && sha256sum ./*.sass) > "${out}/sha256.tsv"
exit "${status}"
