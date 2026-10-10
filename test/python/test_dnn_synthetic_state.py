import copy
import unittest
import torch
from tilemega.dnn.check_generated import synthetic_state

class SyntheticDnnStateTest(unittest.TestCase):
    def test_deterministic_exported_state_with_valid_normalization(self):
        model=torch.nn.Sequential(torch.nn.Conv2d(3,8,3),torch.nn.BatchNorm2d(8)).eval().bfloat16()
        program=torch.export.export(model,(torch.ones(2,3,7,7,dtype=torch.bfloat16),),strict=False)
        module=program.module(); original=copy.deepcopy(module.state_dict())
        state=synthetic_state(module)
        duplicate=copy.deepcopy(module); synthetic_state(duplicate)
        self.assertTrue(all(torch.equal(v,duplicate.state_dict()[n]) for n,v in state.items()))
        self.assertTrue(all(v.dtype==original[n].dtype for n,v in state.items()))
        self.assertTrue(torch.all(state['1.running_var']==1))
        self.assertTrue(torch.all(state['1.running_mean']==0))
        self.assertFalse(torch.equal(state['0.weight'],original['0.weight']))
        output=module(torch.ones(2,3,7,7,dtype=torch.bfloat16))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(output.float().std().item(),0)

if __name__=='__main__':unittest.main()
