"""Replay exported operators from observed native boundaries for diagnosis."""
import torch


@torch.inference_mode()
def local_replay(fp32_module, fp32_args, bf16_module, bf16_args, actual, through):
    if through not in actual:
        raise ValueError('local diagnostic endpoint is not a materialized output: ' + through)
    expected = []
    class StopAtBoundary(Exception):
        pass
    class Replay(torch.fx.Interpreter):
        def __init__(self, module):
            super().__init__(module)
            self.boundaries = {}
        def run_node(self, node):
            value = super().run_node(node)
            if node.name in actual:
                native = actual[node.name]
                if not isinstance(value, torch.Tensor) or native.shape != value.shape:
                    raise ValueError('local diagnostic boundary shape differs: ' + node.name)
                self.boundaries[node.name] = value.detach().float().cpu().clone()
                if node.name == through:
                    raise StopAtBoundary()
                value = native.to(device=value.device, dtype=value.dtype).clone()
            return value
    for module, args in ((fp32_module, fp32_args), (bf16_module, bf16_args)):
        replay = Replay(module)
        try:
            replay.run(*args)
        except StopAtBoundary:
            pass
        else:
            raise ValueError('local diagnostic endpoint was not executed: ' + through)
        expected.append(replay.boundaries)
    if expected[0].keys() != expected[1].keys():
        raise ValueError('FP32/BF16 local boundaries differ')
    rows = []
    for name, fp32 in expected[0].items():
        native, bf16 = actual[name].float().cpu(), expected[1][name]
        per_batch = []
        for index, (a,f,b) in enumerate(zip(native,fp32,bf16)):
            per_batch.append(dict(batch=index, tm_fp32_mse=(a-f).square().mean().item(),
                bf16_fp32_mse=(b-f).square().mean().item(),
                tm_bf16_max_error=(a-b).abs().max().item()))
        rows.append(dict(reference=name, per_batch=per_batch))
    return dict(scope='diagnostic only; each retained boundary starts from TileMega inputs',
                through=through, boundaries=rows)
