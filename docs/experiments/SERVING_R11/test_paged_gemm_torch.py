#!/usr/bin/env python3
"""Reuse R10's BF16 epilogue reference, exercising the page consumer instead."""
import importlib.util
from pathlib import Path

source=Path(__file__).resolve().parents[1]/'SERVING_R10/test_gemm_torch_matrix.py'
spec=importlib.util.spec_from_file_location('gemm_reference',source)
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
reference.ROWS=(1,3,17)
reference.SHAPES=[(64,2048),(256,6144)]
# Stages no longer controls B buffering; these run distinct page groupings.
reference.CONFIGS=reference.CONFIGS[:4]
if __name__=='__main__':
    raise SystemExit(reference.main())
