#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# EX-E3 step 4: does one release fence by thread 0 hold?
#
# Phases, via PHASES=build,sass,preflight,scan:
#   sass       proves each switch reached the code.  Counts alone cannot
#              separate the candidate from the control -- both keep six
#              barriers and two fences -- so the census is paired with a diff
#              of the two bodies
#   preflight  the deps are circular, so a grid above `resident_cap` deadlocks
#              instead of failing; every grid is checked co-resident first and
#              a non-co-resident grid is refused, not silently timed out
#   scan       4 release orders x acquire in {1,0} x grid in {64,128,256} x
#              tile in {1024,4096,16384}, 50 fresh processes each, tallied
#              pass/mismatch/hang/error
#
# Reading the result: `no_fence` is the sensitivity control.  Where it passes,
# the cell cannot detect a missing release at all (F-3, F-10) and the candidate
# arm says nothing there.  Only cells where `no_fence` mismatches are evidence.
#
# The acquire axis exists because the first scan had no such cell.  All four
# arms ran the consumer's `__threadfence()` unconditionally, and that fence
# invalidates the L1 line the stale read needs (F-10) -- so the detector was
# off in all 36 cells and `thread0_fence` holding 50/50 was evidence of
# nothing.  V_A drops the acquire fence alongside the release one in its own
# `no_fence` arm (event_sync.cu:186), which is what makes it sensitive.  A cell
# is readable for the candidate only where `no_fence` mismatches *and*
# `per_writer` still passes: the first says the detector is awake there, the
# second that the reference protocol still holds there.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${OUT_DIR:-${here}/raw_litmus}"
nvcc="${CUDACXX:-/usr/local/cuda-12.8/bin/nvcc}"
# The litmus is a property of the memory model, so it is worth asking on a part
# whose answer may differ; ARCH is what lets the sm_120 runner ask it there.
arch="${ARCH:-sm_89}"
cuobjdump="${CUOBJDUMP:-/usr/local/cuda-12.8/bin/cuobjdump}"
runs="${RUNS:-50}"
timeout_s="${TIMEOUT_S:-30}"
phases="${PHASES:-build,sass,preflight,scan}"
has() { [[ ",${phases}," == *",$1,"* ]]; }

releases=(per_writer thread0_fence no_barrier no_fence)
acquires=(1 0)
grids=(64 128 256)
tiles=(1024 4096 16384)

mkdir -p "${raw}/bin" "${raw}/log"
# A status file that outlives a partial run reads as a pass with no data
# behind it; drop it up front so only this run can write it.
rm -f "${raw}/status.txt"

if has build; then
  "${nvcc}" -O2 -std=c++17 -arch="${arch}" -lineinfo --ptxas-options=-v \
    -I"${repo}/test/harness" -o "${raw}/bin/litmus" "${here}/litmus.cu" \
    2> "${raw}/log/build.ptxas.txt"
  "${nvcc}" -O2 -std=c++17 -arch="${arch}" -cubin -I"${repo}/test/harness" \
    -o "${raw}/bin/litmus.cubin" "${here}/litmus.cu"
fi

if has sass; then
  # per_writer keeps MEMBAR before the release barrier, where every thread runs
  # it; thread0_fence moves it after that barrier and inside the thread-0
  # branch.  Same six barriers, same two fences either way, so the census is
  # not enough on its own and the two bodies are diffed as well.
  dump="${raw}/log/litmus.sass"
  "${cuobjdump}" --dump-sass "${raw}/bin/litmus.cubin" > "${dump}"
  printf 'release\tbar_sync\tmembar\tatom_red\tinsns\n' > "${raw}/census.tsv"
  total="$(wc -l < "${dump}")"
  starts=(); names=()
  while IFS= read -r entry; do
    starts+=("${entry%%:*}")
    names+=("$(printf '%s' "${entry#*Function : }" | tr -d ' \r')")
  done < <(grep -n 'Function : ' "${dump}")
  for ((k=0; k<${#starts[@]}; ++k)); do
    a="${starts[$k]}"
    if ((k + 1 < ${#starts[@]})); then b=$((${starts[$((k + 1))]} - 1)); else b="${total}"; fi
    r="$(sed -n 's/.*litmus_kernelILi\([0-9]\)E.*/\1/p' <<<"${names[$k]}")"
    [[ -n "${r}" ]] || continue
    norm="${raw}/log/r${r}.norm"
    sed -n "${a},${b}p" "${dump}" | grep -oE '^\s+/\*[0-9a-f]{4}\*/\s+.*;' \
      | sed -E 's@^\s*/\*[0-9a-f]{4}\*/\s*@@; s/\s+/ /g' > "${norm}"
    printf '%s\t%s\t%s\t%s\t%s\n' "${r}" \
      "$(grep -c 'BAR\.SYNC' "${norm}" || true)" \
      "$(grep -c 'MEMBAR' "${norm}" || true)" \
      "$(grep -cE '\b(ATOM|RED)' "${norm}" || true)" \
      "$(wc -l < "${norm}")" >> "${raw}/census.tsv"
  done
  cat "${raw}/census.tsv"
  get() { awk -F'\t' -v r="$1" -v c="$2" 'NR>1 && $1==r {print $c}' "${raw}/census.tsv"; }
  [[ "$(wc -l < "${raw}/census.tsv")" -eq 5 ]] || {
    echo "expected 4 arms in the census" >&2; exit 1; }
  # Two fewer, not one: the arm drops the release-side barrier and the
  # acquire-side one.  Keeping the acquire barrier is what made the first
  # scan's negative control pass 50/50 everywhere (litmus.cu, `no_barrier`).
  [[ "$(get 2 2)" -eq "$(( $(get 0 2) - 2 ))" ]] || {
    echo 'no_barrier kept its BAR.SYNC: the switch did not reach the code' >&2; exit 1; }
  [[ "$(get 3 3)" -eq "$(( $(get 0 3) - 1 ))" ]] || {
    echo 'no_fence kept its MEMBAR: the switch did not reach the code' >&2; exit 1; }
  diff -u "${raw}/log/r0.norm" "${raw}/log/r1.norm" \
    > "${raw}/per_writer_vs_thread0.diff" || true
  [[ -s "${raw}/per_writer_vs_thread0.diff" ]] || {
    echo 'thread0_fence compiled identically to per_writer: the candidate arm is the control' >&2
    exit 1; }
  echo "arms distinct; candidate differs in $(grep -c '^[+-][^+-]' \
    "${raw}/per_writer_vs_thread0.diff") instructions"
fi

if has preflight; then
  : > "${raw}/preflight.txt"
  # Static first: the arms must be occupancy-matched, or a difference between
  # them could be residency rather than the release order.  Names come out of
  # the cubin because the mangling is the compiler's, not ours.
  mapfile -t kernels < <("${cuobjdump}" --dump-elf-symbols \
    "${raw}/bin/litmus.cubin" | grep -o '_Z13litmus_kernelIL[^ ]*' | sort -u)
  [[ "${#kernels[@]}" -eq "${#releases[@]}" ]] || {
    echo "expected ${#releases[@]} kernels in the cubin, found ${#kernels[@]}" >&2
    exit 1; }
  ctas=""
  for k in "${kernels[@]}"; do
    line="$("${repo}/tools/tilemega-occupancy" --cubin "${raw}/bin/litmus.cubin" \
      --kernel "${k}" --dynamic-smem 32 | tail -1)"
    echo "${k} ${line}" | tee -a "${raw}/preflight.txt"
    n="$(sed -n 's/.* ctas_per_sm=\([0-9][0-9]*\).*/\1/p' <<<"${line}")"
    limit="$(sed -n 's/.* resident_limit=\([0-9][0-9]*\).*/\1/p' <<<"${line}")"
    [[ -n "${n}" && -n "${limit}" ]] || { echo "no occupancy for ${k}" >&2; exit 1; }
    [[ -z "${ctas}" || "${n}" == "${ctas}" ]] || {
      echo "arms are not occupancy-matched: ${n} vs ${ctas}" >&2; exit 1; }
    ctas="${n}"
    [[ "${limit}" -ge "${grids[-1]}" ]] || {
      echo "resident_limit ${limit} < grid ${grids[-1]}: circular deps would deadlock" >&2
      exit 1; }
  done
  echo "arms occupancy-matched at ctas_per_sm=${ctas}" | tee -a "${raw}/preflight.txt"
  for grid in "${grids[@]}"; do
    line="$("${raw}/bin/litmus" --release per_writer --grid "${grid}" \
      --tile 1024 --iters 1 | grep '^OCCUPANCY')"
    echo "${line}" | tee -a "${raw}/preflight.txt"
    [[ "${line}" == *"co_resident=yes"* ]] || {
      echo "grid ${grid} is not co-resident: circular deps would deadlock" >&2
      exit 1; }
  done
fi

if has scan; then
  printf 'release\tacquire\tgrid\ttile\tpass\tmismatch\thang\terror\truns\n' \
    > "${raw}/litmus.tsv"
  for release in "${releases[@]}"; do
    for acq in "${acquires[@]}"; do
      for grid in "${grids[@]}"; do
        for tile in "${tiles[@]}"; do
          args=(--release "${release}" --grid "${grid}" --tile "${tile}")
          if [[ "${acq}" == 0 ]]; then args+=(--no-acquire-fence); fi
          log="${raw}/log/${release}_a${acq}_g${grid}_t${tile}.txt"; : > "${log}"
          pass=0; mismatch=0; hang=0; error=0
          for ((i=0; i<runs; ++i)); do
            rc=0
            timeout "${timeout_s}s" "${raw}/bin/litmus" "${args[@]}" \
              >> "${log}" 2>&1 || rc=$?
            case "${rc}" in
              0) pass=$((pass + 1)) ;;
              1) mismatch=$((mismatch + 1)) ;;
              124) hang=$((hang + 1)) ;;
              *) error=$((error + 1)) ;;
            esac
          done
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${release}" \
            "${acq}" "${grid}" "${tile}" "${pass}" "${mismatch}" "${hang}" \
            "${error}" "${runs}" | tee -a "${raw}/litmus.tsv"
        done
      done
    done
  done

  # The rule in force today has to hold wherever the protocol is fully present,
  # or a mismatch elsewhere is the harness's fault.  Asserted only at
  # acquire=1: with the acquire fence removed the protocol under test is
  # incomplete, so per_writer failing there is a measurement about that fence,
  # not an unsound harness.
  bad="$(awk -F'\t' 'NR>1 && $1=="per_writer" && $2==1 && $5!=$9' \
    "${raw}/litmus.tsv")"
  [[ -z "${bad}" ]] || {
    echo "per_writer did not pass everywhere at acquire=1; harness is unsound:" >&2
    echo "${bad}" >&2; exit 1; }

  for acq in "${acquires[@]}"; do
    n="$(awk -F'\t' -v a="${acq}" 'NR>1 && $1=="no_fence" && $2==a && $6>0' \
      "${raw}/litmus.tsv" | wc -l)"
    echo "SENSITIVE acquire=${acq} cells=${n}"
  done

  # Evidence about the candidate exists only where the detector is awake and
  # the reference protocol still holds.  Zero such cells is E3-4 failing on
  # completeness -- which is exactly what the first scan did.
  printf 'acquire\tgrid\ttile\tper_writer_pass\tno_fence_mismatch\tthread0_pass\tno_barrier_mismatch\n' \
    > "${raw}/readable.tsv"
  awk -F'\t' 'NR>1 {k=$2"_"$3"_"$4; p[$1"_"k]=$5; m[$1"_"k]=$6; n[k]=$9}
    END { for (k in n) {
      if (m["no_fence_" k] > 0 && p["per_writer_" k] == n[k]) {
        split(k, f, "_")
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", f[1], f[2], f[3],
          p["per_writer_" k], m["no_fence_" k], p["thread0_fence_" k],
          m["no_barrier_" k] } } }' "${raw}/litmus.tsv" \
    | sort -t"$(printf '\t')" -k1,1nr -k2,2n -k3,3n >> "${raw}/readable.tsv"
  det="$(($(wc -l < "${raw}/readable.tsv") - 1))"
  cat "${raw}/readable.tsv"
  echo "READABLE cells=${det}"
  [[ "${det}" -gt 0 ]] || {
    echo 'no readable cell: no_fence never mismatched where per_writer holds' >&2
    echo 'the scan says nothing about thread0_fence (E3-4 incomplete)' >&2
    exit 1; }
fi

# Records which phases ran: a bare PASS after `PHASES=build` would claim the
# whole litmus succeeded when nothing was measured.
echo "PASS phases=${phases}" > "${raw}/status.txt"
