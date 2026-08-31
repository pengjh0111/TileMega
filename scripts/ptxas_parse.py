#!/usr/bin/env python3
"""Parse nvcc/ptxas resource-usage diagnostics into JSON lines."""

import json
import re
import sys


ENTRY = re.compile(r"Function properties for\s+(.+)")
REGS = re.compile(r"Used\s+(\d+)\s+registers")
SMEM = re.compile(r"(?:(\d+) bytes smem|([0-9]+) bytes stack frame)")
SPILL = re.compile(r"(\d+) bytes spill stores,\s*(\d+) bytes spill loads")


def main() -> int:
    text = sys.stdin.read() if len(sys.argv) == 1 else open(sys.argv[1], encoding="utf-8").read()
    records = []
    current = {"function": "unknown", "registers": 0, "smem_bytes": 0,
               "spill_store_bytes": 0, "spill_load_bytes": 0}
    seen = False
    for line in text.splitlines():
        match = ENTRY.search(line)
        if match:
            if seen:
                records.append(current)
            current = {"function": match.group(1).strip(), "registers": 0,
                       "smem_bytes": 0, "spill_store_bytes": 0,
                       "spill_load_bytes": 0}
            seen = True
        if match := REGS.search(line):
            current["registers"] = int(match.group(1)); seen = True
        if match := SMEM.search(line):
            if match.group(1):
                current["smem_bytes"] = int(match.group(1)); seen = True
        if match := SPILL.search(line):
            current["spill_store_bytes"] = int(match.group(1))
            current["spill_load_bytes"] = int(match.group(2)); seen = True
    if seen:
        records.append(current)
    for record in records:
        print(json.dumps(record, sort_keys=True))
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
