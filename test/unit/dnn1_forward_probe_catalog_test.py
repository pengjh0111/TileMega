"""CPU contracts for recipe provenance and generated probe dependencies."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


catalog = module('dnn1_forward_probe_catalog')
recipe = module('dnn1_forward_recipe_probe')


class PortableProbeTest(unittest.TestCase):
    def target(self, root):
        target = root/'target.json'
        section = dict(schema=3, key_layouts=[['a', 'ns', ['m']]],
            keys=[[0, '032'], [0, '64']], median_by_gpu={str(g):[1,2] for g in (0,5,7)},
            calibration_epoch=dict(source_recipes=dict(base='base_sha',tm32='extra_sha')),
            supplements=[dict(raw_sources={'sm90_forward_constants.json':dict(sha256='extra_sha')},
                key_indices=[1])])
        target.write_text(json.dumps(dict(sm90_forward_constants=section)))
        return target, section

    def test_source_ids_and_exact_selector_spelling(self):
        with tempfile.TemporaryDirectory() as name:
            target, section = self.target(Path(name))
            result = catalog.catalog(target)
            self.assertEqual(result['medians'], 6)
            self.assertEqual(result['recipes']['base']['keys'][0]['key'], 'm=032')
            self.assertEqual(result['recipes']['tm32']['keys'][0]['constant_id'], 1)
            section['supplements'][0]['key_indices'] = [1, 1]
            target.write_text(json.dumps(dict(sm90_forward_constants=section)))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                catalog.catalog(target)

    def test_custom_methods_keep_actual_production_dependencies(self):
        base = (ROOT/'scripts/dnn1_constants.cu').read_text()
        for name in recipe.RECIPES:
            source = recipe.generate_source(name, base)
            self.assertEqual(len(recipe.catalog_spans(source)), 2)
            self.assertNotIn('/root/', source)
        matrix = recipe.generate_source('mma32_math', base)
        self.assertIn('#include <tilemega/Backend/Sm90ForwardMma32.cuh>', matrix)
        self.assertIn('Math::Accumulate(b,a,accum)', matrix)
        self.assertIn('Body::Depthwise(halo,weights,panel', recipe.generate_source('dwpw', base))
        gated = recipe.generate_source('gated_dw', base)
        self.assertIn('p.output_stride=Live/2', gated)
        self.assertIn('gated_dw_epilogue_diag_ns', gated)


if __name__ == '__main__':
    unittest.main()
