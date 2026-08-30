#!/bin/bash
# V2-g: grid FIXED at 340; only the first K blocks take part in the handshake.
# Separates "grid size" from "number of concurrent readers".
set -u; cd "$(dirname "$0")/.."; N=${N:-50}
printf "  %-6s %-10s %s\n" K failed/$N "bad-value histogram"
for K in 2 8 32 128 339; do
  fail=0; hist=""
  for i in $(seq 1 $N); do
    o=$(V2_VERIFY_LAST=$K V2_REPS=1 timeout 25 ./v2_host V2_g/v2_g_k$K.cubin v2_g_k$K 340 scalar 1024 1 2>&1)
    grep -q "all slots verified" <<<"$o" || { fail=$((fail+1)); hist+=$(grep -o 'histogram:.*' <<<"$o"|sed 's/histogram://')" "; }
  done
  agg=$(tr ' ' '\n' <<<"$hist"|grep -o '^[0-9][0-9.]*'|sort -n|uniq -c|awk '{printf "%s x%s  ",$2,$1}')
  printf "  %-6s %-10s %s\n" $K "$fail/$N" "$agg"
done
