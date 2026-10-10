import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tilemega.moe.profile_identity import read_profile, verify_profile, main


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def seal(value):
    value['profile_id'] = digest({k: v for k, v in value.items() if k != 'profile_id'})
    return value


def fixture():
    # Two experts, four tokens, selection [0,1,1,1]. Disjoint two-token
    # windows activate two experts then one; these counts are hand enumerated.
    coordinates = {
        '1': dict(tokens=1, windows=4, assignments_per_window=1,
                  distinct_experts_histogram={'1': 4}, expected_distinct_experts=1.,
                  tokens_per_expert_histograms=[{'0': 3, '1': 1}, {'0': 1, '1': 3}]),
        '2': dict(tokens=2, windows=2, assignments_per_window=2,
                  distinct_experts_histogram={'1': 1, '2': 1}, expected_distinct_experts=1.5,
                  tokens_per_expert_histograms=[{'0': 1, '1': 1}, {'1': 1, '2': 1}]),
    }
    for point in coordinates.values():
        point['group_blocks_histograms'] = {
            str(b): copy.deepcopy(point['distinct_experts_histogram']) for b in (16, 32, 64, 128)}
        point['virtual_rows_totals'] = {
            str(b): [4, 0] if point['tokens']==1 else [3, 1, 0] for b in (16, 32, 64, 128)}
        point['virtual_rows_histograms'] = {
            str(b): [{'1': 4}, {}] if point['tokens']==1 else [{'1': 1, '2': 1}, {'1': 1}, {}] for b in (16, 32, 64, 128)}
    sampling = dict(layers=2, experts=2, top_k=1, config_sha256='c'*64,
                    tokens_sha256='d'*64)
    sampling_sha = hashlib.sha256((json.dumps(sampling, indent=2, sort_keys=True)+'\n').encode()).hexdigest()
    common = dict(checkpoint=dict(config_sha256='c'*64, index_sha256='e'*64),
                  implementation=dict(script_sha256='a'*64, hf_model_source_sha256='b'*64,
                                      hf_experts_source_sha256='f'*64,
                                      attention='sdpa', experts='grouped_mm'))
    embedding = dict(copy.deepcopy(common), output_sha256='0'*64,
                     inputs={'/capture/sampling.json': sampling_sha,
                             '/capture/tokens.json': 'd'*64})
    layers = []
    for i in range(2):
        identity = dict(copy.deepcopy(common), layer=i, output_sha256=str(i+1)*64,
                        routing_sha256='3'*64,
                        inputs={'/capture/sampling.json': sampling_sha,
                                f'/capture/hidden_{i:02d}.safetensors': str(i)*64})
        layers.append(dict(layer=i, coordinates=copy.deepcopy(coordinates), identity=identity))
    return seal(dict(schema='tilemega.dm1.routing.profile.v1', evidence='verified',
                     sampling=sampling, embedding_identity=embedding, layers=layers))


class MoeProfileIdentityTest(unittest.TestCase):
    def test_compiler_verification_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/'profile.json', Path(folder)/'identity.json'
            value = fixture();source.write_text(json.dumps(value))
            args = ['--path', str(source), '--output', str(output), '--layers', '2',
                    '--experts', '2', '--top-k', '1', '--tokens', '1,2']
            self.assertEqual(main(args), 0)
            self.assertEqual(json.loads(output.read_text()), dict(profile_id=value['profile_id'],
                file_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            output.unlink();value['profile_id'] = 'a'*64;source.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'content differs'):
                main(args)
            self.assertFalse(output.exists())

    def test_serving_config_profile_is_explicit_and_moe_only(self):
        from tilemega.cli import read_config
        from tilemega.serving.execution import compiler_features
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config = root/'run.json'; model = root/'config.json'
            profile = root/'profile.json'; profile.write_text('{}')
            value = dict(model=dict(path=folder), features=dict(routing_profile=str(profile), moe_profile_layer=24))
            config.write_text(json.dumps(value));model.write_text('{"model_type":"qwen3_moe"}')
            actual = read_config(config)['features']
            self.assertEqual(actual['routing_profile'], str(profile.resolve()))
            self.assertEqual(compiler_features(actual)['moe_profile_layer'], 24)
            model.write_text('{"model_type":"qwen3"}')
            with self.assertRaisesRegex(ValueError, 'requires qwen3_moe'):read_config(config)
            model.write_text('{"model_type":"qwen3_moe"}')
            value['features']['moe_profile_layer'] = -1;config.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'nonnegative integer'):read_config(config)
            value['features'] = {};config.write_text(json.dumps(value))
            self.assertNotIn('routing_profile', read_config(config)['features'])

    def test_virtual_row_conservation(self):
        for corrupted in ([3, 0], [True, 0], [4], [4, 1], [4, -1], [0, 4]):
            value=fixture()
            value['layers'][0]['coordinates']['1']['virtual_rows_totals']['16']=corrupted
            with self.subTest(corrupted=corrupted), self.assertRaises(ValueError):
                self.check(seal(value))

    def test_virtual_row_distribution(self):
        for bins in ({'0': 4}, {'01': 4}, {'2': 4}, {'1': 3}, {'1': True}, {'17': 4}, {'1': 0}):
            value=fixture()
            value['layers'][0]['coordinates']['1']['virtual_rows_histograms']['16'][0]=bins
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                self.check(seal(value))
        value=fixture()
        value['layers'][0]['coordinates']['1']['virtual_rows_histograms']['16'][1]={'1': 1}
        with self.assertRaises(ValueError):
            self.check(seal(value))

    def check(self, value):
        return verify_profile(value, layers=2, experts=2, top_k=1, tokens=(1, 2))

    def test_content_identity_and_file_identity(self):
        value = fixture()
        self.assertEqual(self.check(value), value['profile_id'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'profile.json'
            path.write_text(json.dumps(value, indent=2)+'\n')
            actual, identity = read_profile(path, layers=2, experts=2, top_k=1, tokens=(1, 2))
            self.assertEqual(actual, value)
            self.assertEqual(identity['file_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(identity['profile_id'], value['profile_id'])

    def test_unsealed_modification(self):
        value = fixture()
        value['layers'][0]['coordinates']['1']['expected_distinct_experts'] = 2.
        with self.assertRaisesRegex(ValueError, 'content differs'):
            self.check(value)

    def test_chain_and_reference_mismatch(self):
        for change in ('chain', 'implementation', 'index', 'sampling'):
            with self.subTest(change=change):
                value = fixture()
                layer = value['layers'][1]['identity']
                if change == 'chain':
                    layer['inputs']['/capture/hidden_01.safetensors'] = '5'*64
                elif change == 'implementation':
                    layer['implementation']['hf_model_source_sha256'] = '5'*64
                elif change == 'index':
                    layer['checkpoint']['index_sha256'] = '5'*64
                else:
                    value['sampling']['tokens_sha256'] = '5'*64
                with self.assertRaises(ValueError):
                    self.check(seal(value))

    def test_assignment_conservation(self):
        value = fixture()
        value['layers'][0]['coordinates']['2']['tokens_per_expert_histograms'][1] = {'2': 2}
        with self.assertRaisesRegex(ValueError, 'assignment conservation'):
            self.check(seal(value))

    def test_coordinate_and_model_binding(self):
        value = fixture()
        with self.assertRaisesRegex(ValueError, 'configuration'):
            verify_profile(value, layers=2, experts=16, top_k=1, tokens=(1, 2))
        with self.assertRaisesRegex(ValueError, 'coordinates differ'):
            verify_profile(value, layers=2, experts=2, top_k=1)
        del value['layers'][0]['coordinates']['1']
        with self.assertRaisesRegex(ValueError, 'coordinates differ'):
            self.check(seal(value))

    def test_histogram_types_and_bins(self):
        for modification in ({'01': 4}, {'1': True}, {'1': 4.0}, {'1': 3}, {'3': 4}):
            with self.subTest(modification=modification):
                value = fixture()
                value['layers'][0]['coordinates']['1']['distinct_experts_histogram'] = modification
                with self.assertRaises(ValueError):
                    self.check(seal(value))

    def test_mean_and_nonfinite(self):
        value = fixture()
        value['layers'][0]['coordinates']['2']['expected_distinct_experts'] = 1.
        with self.assertRaisesRegex(ValueError, 'mean differs'):
            self.check(seal(value))
        value['layers'][0]['coordinates']['2']['expected_distinct_experts'] = float('nan')
        with self.assertRaises(ValueError):
            self.check(seal(value))

    def test_joint_block_distribution_and_domain(self):
        value = fixture()
        value['layers'][0]['coordinates']['2']['group_blocks_histograms']['16'] = {'2': 2}
        with self.assertRaisesRegex(ValueError, 'joint block mean'):
            self.check(seal(value))
        value = fixture()
        del value['layers'][0]['coordinates']['2']['group_blocks_histograms']['128']
        with self.assertRaisesRegex(ValueError, 'block coordinates'):
            self.check(seal(value))

    def test_duplicate_json_and_ambiguous_input(self):
        value = fixture()
        value['layers'][0]['identity']['inputs']['/different/hidden_00.safetensors'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.check(seal(value))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'profile.json'
            path.write_text('{"profile_id":"a", "profile_id":"b"}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_profile(path, layers=2, experts=2, top_k=1, tokens=(1, 2))


if __name__ == '__main__':
    unittest.main()
