"""Check the installed CPU torch export adapter without initializing NPU."""
import json
import torch
from tilemega.export_bridge import serialize

class Add(torch.nn.Module):
    def forward(self, x, y):
        return x+y

rows=torch.export.Dim('rows',min=2,max=16)
args=(torch.ones(4,8),torch.ones(4,8))
p=torch.export.export(Add(),args,dynamic_shapes=({0:rows},{0:rows}))
bridge=serialize(p)
assert bridge['guards']==["L['flat_args'][1].size()[0] == L['flat_args'][0].size()[0]"]
assert len(bridge['range_constraints'])==1
assert list(bridge['range_constraints'].values())==['VR[2, 16]']
assert torch.equal(p.module()(*args),args[0]+args[1])

derived=torch.export.export(Add(),args,dynamic_shapes=({0:2*rows},{0:2*rows}))
try:
    serialize(derived)
except RuntimeError as error:
    assert 'derived input dimension' in str(error)
else:
    raise AssertionError('Derived dimensions must fail closed')
initialized=bool(getattr(getattr(torch,'npu',None),'is_initialized',lambda:False)())
assert not initialized
print(json.dumps(dict(status='passed',torch_version=torch.__version__,checks=[
    'shared_symbol_equality','exact_range','exported_execution','reject_derived_dimension'],
    npu_initialized=initialized)))
