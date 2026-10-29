"""The MoE determinism predicate compares storage, including signed zeros."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('moe_check_common', Path(__file__).resolve().parents[1] / 'unit/mx/moe_check_common.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_signed_zero_is_not_bitwise_equal(dtype):
    plus = torch.tensor([0.0], dtype=dtype)
    minus = torch.tensor([-0.0], dtype=dtype)
    assert torch.equal(plus, minus)
    assert not module.equal_bits(torch, plus, minus)
    assert module.equal_bits(torch, minus, minus.clone())


def test_same_nan_bits_and_different_dtypes_are_explicit():
    nan = torch.tensor([float('nan')], dtype=torch.bfloat16)
    assert module.equal_bits(torch, nan, nan.clone())
    assert not module.equal_bits(torch, torch.tensor([1.0]), torch.tensor([1]))
