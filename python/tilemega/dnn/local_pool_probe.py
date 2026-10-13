"""Isolate pooling arithmetic from errors already present in its input.

This is diagnostic evidence, never a replacement for the fixed DNN gates.
Only per-channel means are retained; full native intermediate tensors are not.
"""


class PoolProbe:
    def __init__(self, nodes):
        self.inputs = {}
        for node in nodes:
            if (node['target'] == 'aten.adaptive_avg_pool2d.default'
                    and len(node.get('shape', [])) == 4
                    and list(map(str, node['shape'][-2:])) == ['1', '1']
                    and len(node['inputs']) == 1):
                self.inputs[node['name']] = node['inputs'][0]
        self.means, self.outputs = {}, {}

    def observe(self, name, value):
        import torch
        if name in self.inputs.values():
            # A CPU FP64 reference avoids borrowing either GPU reduction order.
            self.means[name] = value.detach().cpu().double().mean(
                dim=(-2, -1), keepdim=True).to(torch.bfloat16).float()
        if name in self.inputs:
            self.outputs[name] = value.detach().cpu().float().clone()

    def report(self):
        rows = []
        for output, source in self.inputs.items():
            if source not in self.means or output not in self.outputs:
                rows.append(dict(output=output, input=source, observed=False))
                continue
            actual, expected = self.outputs[output], self.means[source]
            if actual.shape != expected.shape:
                raise ValueError('native global pool input/output shapes disagree: ' + output)
            samples = []
            for batch in range(len(actual)):
                error = actual[batch] - expected[batch]
                samples.append(dict(batch=batch, mse=error.square().mean().item(),
                    max_error=error.abs().max().item(),
                    rounded_disagreements=int((error != 0).sum()),
                    mse_against_each_input_image=[
                        (actual[batch] - image).square().mean().item() for image in expected]))
            rows.append(dict(output=output, input=source, observed=True, per_batch=samples))
        return dict(reference='BF16-rounded CPU FP64 mean of the materialized TileMega input',
                    scope='diagnostic only; differences may include reduction association',
                    operators=rows)
