#!/bin/bash
# V2-e: events AND data both distributed. 50 FRESH processes per grid.
# Fresh processes matter: the defect is a first-launch-after-cuModuleLoad
# phenomenon, so looping reps inside one process hides it (see SUMMARY.md).
set -u
cd "$(dirname "$0")/.."
RUNS=${RUNS:-50}
printf "%-6s %-8s %-14s %s\n" grid runs "failed/runs" "bad-value histogram"
for g in 8 30 80 120 170 340; do
  fail=0; hist=""
  for i in $(seq 1 $RUNS); do
    o=$(V2_REPS=1 ./v2_host V2_e/v2_e.cubin v2_e $g v2e 1024 0 2>&1)
    if ! grep -q "all slots verified" <<<"$o"; then
      fail=$((fail+1)); hist+=$(grep -o 'histogram:.*' <<<"$o" | sed 's/histogram://')" "
    fi
  done
  agg=$(tr ' ' '\n' <<<"$hist" | grep -o '^[0-9.]*' | sort -n | uniq -c | awk '{printf "%s x%s  ",$2,$1}')
  printf "%-6s %-8s %-14s %s\n" $g $RUNS "$fail/$RUNS" "$agg"
done
