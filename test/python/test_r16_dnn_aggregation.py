"""Do not merge distinct native implementations or repeat one measured process."""
import copy
import json
import tempfile
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from summarize_r16 import aggregate_dnn, dnn_runtime_adjustment, dnn_device_identity


class DnnAggregation(unittest.TestCase):
    def rows(self, candidate, artifact):
        return [dict(model='mbv1', system='tilemega', batch=1, precision='bf16',
            compile_mode='', candidate_id=candidate, candidate_receipt_sha256=candidate,
            native_artifact_ids=artifact, round=r, pid=100+r, gpu=6,
            median_ms=1+r/10, p90_ms=2+r/10, exported_program_sha256='same-checkpoint',
            acceptance='provisional_container_pid_namespace') for r in range(3)]

    def test_distinct_bodies_and_binaries_stay_separate(self):
        rows=aggregate_dnn(self.rows('mma', 'binary-a')+self.rows('wgmma', 'binary-b'))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['protocol_complete'] for row in rows))
        self.assertEqual({row['native_artifact_ids'] for row in rows}, {'binary-a','binary-b'})

    def test_repeated_process_or_round_is_rejected(self):
        rows=self.rows('mma', 'binary-a')
        for change in ('pid', 'round'):
            broken=copy.deepcopy(rows)
            broken[1][change]=broken[0][change]
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate_dnn(broken)

    def test_checkpoint_variants_have_separate_complete_protocols(self):
        original=self.rows('mma', 'binary-a')
        changed=copy.deepcopy(original)
        for row in changed:
            row['exported_program_sha256']='nonzero-residual-checkpoint'
            row['source_checkpoint_sha256']='checkpoint-b'
            row['pid']+=10
        cells=aggregate_dnn(original+changed)
        self.assertEqual(len(cells), 2)
        self.assertTrue(all(cell['protocol_complete'] for cell in cells))

    def test_optional_checkpoint_hash_does_not_split_identical_exports(self):
        rows=self.rows('mma', 'binary-a')
        rows[1]['source_checkpoint_sha256']='checkpoint-a'
        rows[2]['source_checkpoint_sha256']='checkpoint-a'
        cell,=aggregate_dnn(rows)
        self.assertTrue(cell['protocol_complete'])
        self.assertEqual(cell['source_checkpoint_sha256'], 'checkpoint-a')
        self.assertEqual(cell['checkpoint_hash_processes'], 2)
        rows[2]['source_checkpoint_sha256']='checkpoint-b'
        with self.assertRaises(ValueError):
            aggregate_dnn(rows)

    def test_same_index_different_physical_cards_never_complete_one_cell(self):
        rows = self.rows('mma', 'binary-a')
        for index, row in enumerate(rows):
            row['physical_gpu_uuid'] = 'old-card' if index < 2 else 'new-card'
            row['physical_gpu_identity_status'] = 'verified'
        cells = aggregate_dnn(rows)
        self.assertEqual(len(cells), 2)
        self.assertFalse(any(c['protocol_complete'] for c in cells))

    def test_unknown_index_only_is_not_verified_same_card(self):
        cell, = aggregate_dnn(self.rows('mma', 'binary-a'))
        self.assertTrue(cell['protocol_complete'])
        self.assertFalse(cell['same_physical_gpu_protocol_complete'])
        self.assertEqual(cell['physical_gpu_identity_status'], 'unknown')

    def test_distinct_boots_remain_separate_even_with_same_card(self):
        rows = self.rows('mma', 'binary-a')
        for index, row in enumerate(rows):
            row.update(physical_gpu_uuid='one-card', hardware_boot_id=str(index // 2))
        cells = aggregate_dnn(rows)
        self.assertEqual(len(cells), 2)

    def test_engine_actual_uuid_requires_no_inventory_guess(self):
        value = {'configuration': {'engine_cache': {'identity': {'gpu_uuid': 'GPU-ABC'}}}}
        identity = dnn_device_identity(value, Path('/nonexistent-r16-proof'))
        self.assertEqual(identity['physical_gpu_uuid'], 'abc')
        self.assertEqual(identity['physical_gpu_identity_status'], 'verified')

    def test_wrapper_probe_is_attributed_without_faking_measurement_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            receipt = directory/'physical-device-identity.json'
            identity = dict(cuda_uuid='abc', nvml_uuid='GPU-abc', probe_exit_code=0,
                            probe_pid=1, wrapper_pid=2, boot_id='boot')
            receipt.write_text(json.dumps(identity))
            measurement = directory/'measurements/measurements.json'
            command = directory/'physical-device-command.json'
            command.write_text(json.dumps(dict(actual_returncode=0, measurements=[str(measurement)])))
            value = dict(pid=3, physical_device_identity=identity,
                         physical_device_identity_file=str(receipt),
                         physical_device_identity_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
            self.assertEqual(dnn_device_identity(value, directory)['physical_gpu_uuid'], 'abc')
            receipt.write_text('{}')
            with self.assertRaises(ValueError):
                dnn_device_identity(value, directory)

    def test_temporary_launch_configuration_stays_explicit(self):
        metadata = dict(enabled=True, temporary_workaround=True, artifact_id='binary-a',
                        library_sha256='library', driver_sha256='driver', production_fix_commit='fix')
        value = dict(artifact_id='binary-a', identity={'binary_sha256':'library'},
                     runtime_shared_memory_optin_workaround=metadata,
                     configuration={'runtime_shared_memory_optin_workaround':metadata})
        adjustment = dnn_runtime_adjustment(value)
        rows = self.rows('mma-optin', 'binary-a')
        for row in rows:
            row['runtime_adjustment'] = adjustment
        cell, = aggregate_dnn(rows)
        self.assertTrue(cell['temporary_workaround'])
        self.assertEqual(cell['runtime_adjustment'], adjustment)
        rows[1]['runtime_adjustment'] = ''
        with self.assertRaises(ValueError):
            aggregate_dnn(rows)
        value['runtime_shared_memory_optin_workaround'] = dict(metadata, library_sha256='different')
        with self.assertRaises(ValueError):
            dnn_runtime_adjustment(value)


if __name__ == '__main__':
    unittest.main()
