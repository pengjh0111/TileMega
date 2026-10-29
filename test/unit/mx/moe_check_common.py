"""Exact storage comparison for native MoE diagnostic fixtures."""


def equal_bits(torch, left, right):
    return (left.shape == right.shape and left.dtype == right.dtype and
        torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))
