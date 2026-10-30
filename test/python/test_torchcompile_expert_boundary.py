"""The optional compile boundary must preserve the HF eager expert operation."""
import importlib.util
from pathlib import Path

import pytest
import torch


def _baseline():
    path = Path(__file__).resolve().parents[2] / 'python/tilemega/serving/torchcompile_baseline.py'
    spec = importlib.util.spec_from_file_location('independent_torch_baseline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boundary_preserves_hf_experts_and_compiles_surrounding_graph():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    config = Qwen3MoeConfig(hidden_size=16, moe_intermediate_size=32,
                           num_experts=4, num_experts_per_tok=2)
    config._experts_implementation = 'eager'
    torch.manual_seed(20260925)
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.mlp = torch.nn.Module()
    model.layer.mlp.experts = Qwen3MoeExperts(config).to(torch.bfloat16).eval()
    for weight in model.parameters():
        torch.nn.init.normal_(weight, std=.02)
    h = torch.randn(3, 16, dtype=torch.bfloat16)
    index = torch.tensor([[0, 1], [2, 3], [1, 2]])
    weights = torch.tensor([[.6, .4]] * 3)
    original = model.layer.mlp.experts.forward
    expected = original(h, index, weights)
    old_limit = torch._dynamo.config.recompile_limit
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    try:
        meta = _baseline().configure_eager_expert_boundary(model, torch)
        def forward(x):
            return model.layer.mlp.experts(x + 0, index, weights) * 1
        compiled = torch.compile(forward, backend=backend, fullgraph=False)
        assert torch.equal(compiled(h), expected)
        assert torch.equal(compiled(h), expected)
        assert graphs, 'surrounding decode operations must actually be captured'
        assert meta['eager_expert_modules'] == ['layer.mlp.experts']
        assert model.layer.mlp.experts.forward.__wrapped__ == original
    finally:
        torch._dynamo.config.recompile_limit = old_limit
        torch._dynamo.reset()


def test_boundary_rejects_a_dense_model():
    with pytest.raises(ValueError, match='requires stacked HF MoE experts'):
        _baseline().configure_eager_expert_boundary(torch.nn.Linear(2, 2), torch)
