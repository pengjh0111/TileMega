"""The full decoder smoke retains original checkpoint FQNs and expert order."""
import unittest
import torch
from tilemega.moe.check_decoder import individual_experts


class DecoderCheckpoint(unittest.TestCase):
    def test_individual_original_fqns(self):
        prefix = 'model.layers.1.mlp.experts.'
        gate_up = torch.arange(3*8*5).reshape(3,8,5).bfloat16()
        down = torch.arange(3*5*4).reshape(3,5,4).bfloat16()
        embedding = torch.zeros(11,5,dtype=torch.bfloat16)
        result = individual_experts({prefix+'gate_up_proj':gate_up,
            prefix+'down_proj':down,'model.embed_tokens.weight':embedding})
        self.assertEqual(len(result),10)
        self.assertIs(result['model.embed_tokens.weight'],embedding)
        for expert in range(3):
            self.assertTrue(torch.equal(result[prefix+f'{expert}.gate_proj.weight'],gate_up[expert,:4]))
            self.assertTrue(torch.equal(result[prefix+f'{expert}.up_proj.weight'],gate_up[expert,4:]))
            self.assertTrue(torch.equal(result[prefix+f'{expert}.down_proj.weight'],down[expert]))


if __name__ == '__main__':
    unittest.main()
