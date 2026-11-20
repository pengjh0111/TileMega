"""Use the MPK Llama demo's bounded output tiles for its SM80 8B graph."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python/tilemega/serving'))
import mpk_baseline as mpk


def bounded_grid(size, demo_grid):
    groups = demo_grid(size)
    if size // groups <= 256:
        return groups
    # The demo uses 256 columns for large projections. Apply the limit to
    # the chosen partition, including when its 96-way partition is illegal.
    if size % 256:
        raise mpk.Unsupported(f'oversized MPK Llama projection cannot use 256-column tiles: {size}')
    return size // 256


class Llama8Engine(mpk.Engine):
    def __init__(self, args, arch, world=1, rank=0):
        super().__init__(args, arch, world, rank)
        config = self.model.config
        if (arch, world, config.hidden_size, config.intermediate_size) != ('LlamaForCausalLM',1,4096,14336):
            raise ValueError('expected the single-GPU Llama-3.1-8B configuration')
        original_grid = self.llama_demo.grid_for_rmsnorm_linear_layer
        self.llama_demo.grid_for_rmsnorm_linear_layer = lambda width: bounded_grid(width,original_grid)


if __name__ == '__main__':
    mpk.Engine = Llama8Engine
    mpk.guarded_main(mpk.main)
