#!/bin/bash
# V2-h: grid FIXED at 170; handshake payload = P chunks (P*1024 floats).
set -u; cd "$(dirname "$0")/.."; N=${N:-50}
printf "  %-6s %-9s %-10s %s\n" P expect failed/$N "top bad values"
for P in 1 4 16 64 256; do
  exp=$((P*1024)); fail=0; hist=""
  for i in $(seq 1 $N); do
    o=$(V2_REPS=1 timeout 25 ./v2_host V2_h/v2_h_p$P.cubin v2_h_p$P 170 scalar $exp 1 2>&1)
    grep -q "all slots verified" <<<"$o" || { fail=$((fail+1)); hist+=$(grep -o 'histogram:.*' <<<"$o"|sed 's/histogram://')" "; }
  done
  agg=$(tr ' ' '\n' <<<"$hist"|grep -o '^[0-9][0-9.]*'|sort -n|uniq -c|sort -rn|head -3|awk '{printf "%s x%s  ",$2,$1}')
  printf "  %-6s %-9s %-10s %s\n" $P $exp "$fail/$N" "$agg"
done
