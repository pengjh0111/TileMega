#!/bin/bash
# V2-d': V1-d with the two section-0.5 bugs fixed, each in isolation, so the
# contribution of each can be attributed. 50 fresh processes per cell.
#   tokfix  : token=%chain -> token=%tk  (the loop-carried token was dead)
#   weakfix : data store weak -> relaxed device (spec 7.2)
# V1-d: 4 producers, consumer bx waits flag[bx*32], reduces 64 chunks = 65536.0
set -u
cd "$(dirname "$0")/.."
N=${N:-50}
for g in 170 340; do
  echo "grid=$g"
  for v in v2_d_orig v2_d_tokfix v2_d_weakfix v2_d_both; do
    fail=0; hist=""
    for i in $(seq 1 $N); do
      o=$(V2_REPS=1 timeout 25 ./v2_host V2_d_fixed/$v.cubin $v $g scalar 65536 4 2>&1)
      grep -q "all slots verified" <<<"$o" || { fail=$((fail+1)); hist+=$(grep -o 'histogram:.*' <<<"$o"|sed 's/histogram://')" "; }
    done
    agg=$(tr ' ' '\n' <<<"$hist"|grep -o '^[0-9][0-9.]*'|sort -n|uniq -c|sort -rn|head -4|awk '{printf "%s x%s  ",$2,$1}')
    printf "  %-14s %-9s %s\n" "$v" "$fail/$N" "$agg"
  done
done
