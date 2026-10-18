import copy
import json
from pathlib import Path
import tempfile
import unittest

import run_r16_hopper_candidates as candidates


class ConfigurationContract(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        path = Path(self.folder.name) / 'layout.json'
        recipe = dict(kind='tile_pages', tile_n=128, tile_k=64,
                      source=dict(kind='weight', name='projection'))
        path.write_text(json.dumps(dict(buffers=[dict(recipe=recipe)])))
        geometry = [dict(index=0, impl='wgmma', tile_m=16, tile_n=64,
                         tile_k=64, stages=4, split_k=2)]
        self.candidate = dict(snapshot_source='frozen', snapshot_head='head',
                              compiler_sha256='compiler', arm='mma', gemm_count=1,
                              layout=str(path), pg_by_phase=dict(decode='pages'),
                              geometry_by_phase=dict(decode=copy.deepcopy(geometry)))
        self.identity = dict(source=dict(root='frozen', head='head', compiler_sha256='compiler'),
                             arch='sm_90', nvcc_arch='sm_90a',
                             defines=dict(TILEMEGA_ARCH_ID=900),
                             caps=dict(wgmma=True, tma=True, bulk_copy=True), trace=False,
                             implementations=dict(gemms=copy.deepcopy(geometry),
                                                  attention=dict(impl='mma16', page_layout='chunk16')),
                             plan=dict(target_spec=dict(arch_tag='sm_90'), seed_arch='native',
                                       seed_target_arch='sm_90', arch_paths='auto', pg='pages',
                                       gemms=geometry, buffers=[dict(recipe=recipe)]))

    def validate(self):
        candidates.assert_implementation(self.identity, self.candidate, 'decode')

    def test_paged_geometry_can_use_subtiles_of_the_shared_packing(self):
        self.validate()

    def test_wrong_execution_path_is_rejected(self):
        self.identity['plan']['pg'] = 'off'
        with self.assertRaisesRegex(ValueError, 'transport or executor'):
            self.validate()

    def test_geometry_must_match_both_manifest_views(self):
        self.identity['implementations']['gemms'][0]['split_k'] = 1
        with self.assertRaisesRegex(ValueError, 'per-GEMM geometry'):
            self.validate()

    def test_packing_is_still_checked(self):
        self.identity['plan']['buffers'][0]['recipe']['tile_k'] = 128
        with self.assertRaisesRegex(ValueError, 'packed weights'):
            self.validate()

    def test_mixed_policy_cannot_hide_missing_hopper_capabilities(self):
        self.identity['caps']['wgmma'] = False
        with self.assertRaisesRegex(ValueError, 'capability'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
