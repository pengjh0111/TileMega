#!/usr/bin/env bash
# ctest wrapper for tools/probes/maca: skip (77) without a MetaX device or
# outside the GPU guard, so plain `ctest` never touches a shared card.
set -u
if [ "${TILEMEGA_GPU_LOCK_HELD:-0}" != "1" ] || ! command -v mx-smi >/dev/null 2>&1; then
  echo "skipped: run ctest through the GPU guard on a MetaX device"; exit 77
fi
exec "$@"
