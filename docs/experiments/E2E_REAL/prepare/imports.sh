#!/bin/bash
# epoch  hoist_imports  control_imports
while :; do
  h=$(grep -c IMPORT_DEGRADED /root/r7_work/llama_hoist/solve.log 2>/dev/null || echo 0)
  c=$(grep -c IMPORT_DEGRADED /tmp/r7-c1b-base.log 2>/dev/null || echo 0)
  printf '%s\t%s\t%s\n' "$(date +%s)" "$h" "$c" >> /tmp/r7-import-rate.tsv
  ps -p 1879731 >/dev/null 2>&1 || break
  sleep 30
done
