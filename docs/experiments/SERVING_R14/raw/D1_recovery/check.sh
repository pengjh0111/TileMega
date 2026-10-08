#!/usr/bin/env bash
set -euo pipefail
cd /root/TileMega
cmake --build build-phase12 --target tilemega tilemega-unit -j 6
ctest --test-dir build-phase12 -R '^(serving_search_rejection|skeleton_search_isolation|stage_flow)$' --output-on-failure
/root/venvs/tilemega-torch213-cu126/bin/python python/tilemega/fingerprint.py --check build-phase12/tools/tilemega
