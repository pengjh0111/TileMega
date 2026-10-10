"""Reproduce the synthetic graph for DNN structural-search host checks."""
import json
import sys
from pathlib import Path

import torch
from torch import nn
from tilemega.export_bridge import serialize

upstream = Path(__file__).resolve().parents[3] / 'python/tilemega/dnn/models/_upstream'
sys.path.insert(0, str(upstream))
from basicsr.models.archs.arch_util import LayerNorm2d


class StructureGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.intro = nn.Conv2d(3, 8, 1)
        self.dw1 = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.pw1 = nn.Conv2d(8, 16, 1)
        self.norm = LayerNorm2d(16)
        self.middle = nn.Conv2d(16, 16, 1)
        self.dw2 = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 16, 1)
        self.end = nn.Conv2d(16, 16, 1)

    def forward(self, x):
        x = self.pw1(nn.functional.hardtanh(self.dw1(self.intro(x)), 0., 6.))
        normalized = self.norm(x)
        branch = self.pw2(nn.functional.hardtanh(self.dw2(self.middle(normalized)), 0., 6.))
        return self.end(branch) + normalized


if __name__ == '__main__':
    torch.manual_seed(20261010)
    model = StructureGraph().eval().to(torch.bfloat16)
    inputs = (torch.randn(2, 3, 7, 9, dtype=torch.bfloat16),)
    exported = torch.export.export(model, inputs, strict=False,
        dynamic_shapes=({0: torch.export.Dim('batch', min=1, max=64)},))
    Path(sys.argv[1]).write_text(json.dumps(serialize(exported), indent=2) + '\n')
    if len(sys.argv) > 2:
        torch.export.save(exported, sys.argv[2])
