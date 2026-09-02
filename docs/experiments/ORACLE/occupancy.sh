#!/usr/bin/env bash
# Does a closed form predict the harness's own ctas_per_sm?
#
# Reads the sweep's screening table and its ptxas logs; writes one row per
# passing candidate and a summary of how often each term binds. No GPU, no
# compile -- pure re-analysis of `raw/`, so it is safe to re-run.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
raw="${here}/raw"
model="${1:-gqa2}"
out="${raw}/occupancy_${model}.tsv"

printf 'config\tsmem\tctas_per_sm\tl1_registers\n' > "${out}"
awk -F'\t' 'NR>1 && $9=="PASS" {print $1"x"$2"x"$3"s"$4"k"$5"\t"$11"\t"$12}' \
    "${raw}/screen_${model}.tsv" |
while read -r tag smem ctas; do
  regs=$(awk '/Compiling entry function.*l1_kernel/{f=1}
              f && /Used [0-9]+ registers/{print $5; exit}' \
         "${raw}/log/${model}_${tag}.ptxas")
  printf '%s\t%s\t%s\t%s\n' "${tag}" "${smem}" "${ctas}" "${regs}" >> "${out}"
done

# RTX 4090 / sm_89: 65536 registers and 1536 threads per SM, 101376 B of
# opt-in shared memory, 256 threads per CTA. Registers are allocated per warp
# at a granularity of 256, which is why `ceil(regs*32/256)*256` and not
# `regs*32` -- 82 registers costs the same as 88 and loses a CTA that the
# unrounded arithmetic would keep.
awk -F'\t' 'NR>1 {
  smem=$2+0; regs=$4+0;
  warp=int((regs*32+255)/256)*256;
  reg_lim=int(65536/(8*warp)); smem_lim=int(101376/smem); thread_lim=6;
  p=reg_lim; if (smem_lim<p) p=smem_lim; if (thread_lim<p) p=thread_lim;
  if (p<1) p=1;
  total++;
  if (p==$3) agree++; else { miss++; if (miss<=10) print "MISS", $1, "smem", smem, "regs", regs, "pred", p, "obs", $3 }
  if (reg_lim==p && smem_lim>p) rb++;
  else if (smem_lim==p && reg_lim>p) sb++;
  else if (reg_lim==p && smem_lim==p) tie++;
  if (thread_lim==p && reg_lim>p && smem_lim>p) tc++;
  s_only=smem_lim; if (s_only>6) s_only=6; if (s_only!=$3) sm++;
  r_only=reg_lim;  if (r_only>6) r_only=6; if (r_only!=$3) rm++;
} END {
  printf "candidates\t%d\nmatch\t%d\nmiss\t%d\n", total, agree, miss+0;
  printf "register_bound\t%d\nsmem_bound\t%d\nboth_bind\t%d\nthread_capped\t%d\n", rb+0, sb+0, tie+0, tc+0;
  printf "smem_term_alone_wrong\t%d\nregister_term_alone_wrong\t%d\n", sm+0, rm+0;
}' "${out}" | tee "${raw}/occupancy_${model}_summary.txt"
