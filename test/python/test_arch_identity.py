"""Keep logical capability tags separate from compiler instruction targets."""
import unittest
from tilemega.build.arch import architecture_identity


class ArchitectureIdentity(unittest.TestCase):
    def test_all_supported_architectures(self):
        for logical, compiled in (('sm_80','sm_80'),('sm_89','sm_89'),
                                  ('sm_90','sm_90a'),('sm_100','sm_100a'),('sm_120','sm_120')):
            options=['-gencode=arch=compute_'+compiled[3:]+',code='+compiled]
            with self.subTest(logical=logical):
                self.assertEqual(architecture_identity(options,{'arch_tag':logical}), (logical,compiled))
                self.assertEqual(architecture_identity(options,{}),(logical,compiled))

    def test_explicit_target_override_and_mismatch(self):
        self.assertEqual(architecture_identity(['-arch=sm_90'],{'arch_tag':'sm_90','nvcc_arch':'sm_90'}),
                         ('sm_90','sm_90'))
        for options in (['-arch=sm_89'],['-arch=sm_90'],[]):
            with self.subTest(options=options),self.assertRaises(ValueError):
                architecture_identity(options,{'arch_tag':'sm_90','nvcc_arch':'sm_90a'})


if __name__=='__main__':unittest.main()
