import tempfile
from pathlib import Path
import unittest
import torch
from safetensors.torch import load_file
from tilemega.dnn.export import write_checkpoint


class DnnCheckpoint(unittest.TestCase):
    def test_fqn_aliases_dtypes_and_values(self):
        module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.BatchNorm1d(8)).eval().bfloat16()
        module.add_module('alias', module[0])
        module[1].num_batches_tracked.fill_(37)
        original = {name: value.clone() for name, value in module.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            metadata = write_checkpoint(module, directory)
            values = load_file(str(Path(directory) / 'model.safetensors'))
            self.assertEqual(set(values), set(original))
            self.assertEqual(set(metadata['tensors']), set(original))
            self.assertEqual(metadata['bytes'], Path(metadata['path']).stat().st_size)
            for name, expected in original.items():
                self.assertEqual(values[name].dtype, expected.dtype)
                self.assertTrue(torch.equal(values[name], expected))
                self.assertTrue(torch.equal(module.state_dict()[name], expected))
            self.assertEqual(values['1.num_batches_tracked'].item(), 37)


if __name__ == '__main__':
    unittest.main()
