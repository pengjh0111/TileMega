import unittest

from tilemega.serving.host_preparation import artifact_binding, validate_artifact_binding


class HostPreparationTest(unittest.TestCase):
    def test_rejects_a_different_prefill_with_the_same_batch(self):
        def identity(name):
            return dict(binary_sha256=f'{name}-binary', manifest_sha256=f'{name}-manifest',
                        source=dict(source_digest=f'{name}-source'))

        config = {'model_type': 'mistral', 'hidden_size': 4096}
        prefill, decode = identity('mistral-prefill'), identity('mistral-decode')
        record = dict(binding=dict(artifact_binding(config, 16, 1088, (prefill, decode)), abi='packed'))
        validate_artifact_binding(record, config, 16, 1088, (prefill, decode))
        with self.assertRaisesRegex(ValueError, 'different plan or workload'):
            validate_artifact_binding(record, config, 16, 1088, (identity('llama-prefill'), decode))


if __name__ == '__main__':
    unittest.main()
