"""Use the Ampere GEMM's valid column partitions for the frozen MPK dense graph."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python/tilemega/serving'))
import mpk_baseline as mpk

default_linear = mpk.grid_for_linear_layer
default_gate = mpk.grid_for_rmsnorm_layer


def ampere_grid(size, groups):
    # Four N warps consume one 64-column atom. Keep the demo's legal partitions.
    if groups and size % groups == 0 and (size // groups) % 64 == 0:
        return groups
    if size % 64:
        raise mpk.Unsupported(f'Ampere MPK linear output width {size} has no full 64-column atom')
    return size // 64


def linear_grid(size, with_residual=False):
    return ampere_grid(size, default_linear(size, with_residual))


def gate_grid(size):
    # Weight shuffling, projection and SiLU share the gate/up partition.
    return ampere_grid(size, default_gate(size))


if __name__ == '__main__':
    mpk.grid_for_linear_layer = linear_grid
    mpk.grid_for_rmsnorm_layer = gate_grid
    mpk.guarded_main(mpk.main)
