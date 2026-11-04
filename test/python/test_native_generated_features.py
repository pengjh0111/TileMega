"""Generate current native coordinates and certify the actual typed calls."""
import copy
import json
import lzma
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tilemega.build.feature_contract import build_defines, validate_gemm_dispatch
from check_sm90_moe_route_contract import check as check_route
from check_sm90_moe_combine_contract import check as check_combine

ROOT = Path(__file__).resolve().parents[2]
COMPILER = os.environ.get('TILEMEGA_BIN')


@unittest.skipUnless(COMPILER, 'native generation requires TILEMEGA_BIN')
class NativeGeneratedFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='tilemega-native-features-')
        cls.root = Path(os.environ.get('TILEMEGA_NATIVE_TEST_OUT', cls.temporary.name))
        cls.root.mkdir(parents=True, exist_ok=True)
        for model in ('qwen3_1_7b', 'qwen3_30b_a3b'):
            source = ROOT / f'configs/serving/sm90d1/inputs/{model}/decode.export.json.xz'
            graph = json.loads(lzma.decompress(source.read_bytes()))
            (cls.root / f'{model}.json').write_text(json.dumps(graph))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def coordinates(self, model, batch):
        family = 'sm90r3' if model == 'qwen3_1_7b' else 'sm90r5/ref_r4'
        value = json.loads((ROOT / f'configs/serving/{family}/{model}_b{batch}.json').read_text())
        return value.get('coordinates', value)

    def generate(self, model, batch, name, coordinate):
        directory = self.root / f'{model}-B{batch}-{name}'
        directory.mkdir()
        config = directory / 'coordinates.json'; config.write_text(json.dumps(coordinate))
        source = directory / 'decode.cu'
        command = [COMPILER, 'compile', str(self.root / f'{model}.json'), str(source),
                   '--tmexec-config', str(config), '--serving', 'decode', '--emit', 'serving',
                   '--batch', str(batch), '--runtime-target', str(ROOT / 'configs/targets/sm_90_h800.json')]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        (directory / 'generate.log').write_text(result.stdout)
        self.assertEqual(result.returncode, 0, result.stdout)
        text = source.read_text()
        manifest = json.loads(Path(str(source)+'.plan.json').read_text())
        self.assertEqual(len(validate_gemm_dispatch(manifest, build_defines(text, []), text)),
                         len(manifest['gemms']))
        corrupted = copy.deepcopy(manifest)
        corrupted['native_dense']['gemms']['qkv']['tile_n'] *= 2
        with self.assertRaises(ValueError):
            validate_gemm_dispatch(corrupted, build_defines(text, []), text)
        return source, text

    def test_dense_reducer_realizations(self):
        for batch in (1, 16):
            with self.subTest(batch=batch):
                c = self.coordinates('qwen3_1_7b', batch)
                self.generate('qwen3_1_7b', batch, 'frozen', c)
                c.update(schema=8, tp=1, argmax='stage', emb='stage', group_wait=False)
                c['attention']['merge'] = 'stage'
                _, text = self.generate('qwen3_1_7b', batch, 'aggregate', c)
                self.assertIn('Argmax<', text)
                self.assertEqual('AttentionMerge<' in text, c['attention']['chunks'] > 1)

    def test_moe_routing_fusion_and_combine(self):
        for batch in (1, 16):
            for mode in ('local', 'la', 'stage', 'unfused', 'aggregate'):
                with self.subTest(batch=batch, mode=mode):
                    c = self.coordinates('qwen3_30b_a3b', batch)
                    c.update(schema=9, tp=1, argmax='la', route=mode if mode in ('local','la','stage') else 'la',
                             combine='la', expert_fuse=mode not in ('unfused','aggregate'))
                    if mode == 'aggregate':
                        c.update(route='stage', combine='stage', argmax='stage', emb='stage', group_wait=False)
                        c['attention']['merge'] = 'stage'
                    source, text = self.generate('qwen3_30b_a3b', batch, mode, c)
                    self.assertEqual('MoeRoute<' in text, c['route'] == 'stage')
                    self.assertEqual('MoeFused<' in text, c['expert_fuse'])
                    self.assertEqual('MoeCombine<' in text, c['combine'] == 'stage')
                    if mode in ('la','stage'):
                        check_route(source)
                    if mode == 'aggregate':
                        check_combine(source)


if __name__ == '__main__':
    unittest.main()
