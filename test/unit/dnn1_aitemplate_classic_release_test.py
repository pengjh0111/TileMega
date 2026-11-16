"""Classic release binds reused receipts and all immutable assets without TVM gates."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import dnn1_aitemplate_classic_release as release
import dnn1_aitemplate_cell as cell


class ClassicReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity = dict(source='fixture', recipe=dict(precision='bf16', graph_mode=True))
        patcher = patch.object(release.runner, 'source_identity', side_effect=lambda: self.identity)
        patcher.start()
        self.addCleanup(patcher.stop)
        proto = self.root / 'protocol.py'
        proto.write_text('fixed protocol')
        self.sources = {str(proto): release.runner.sha(proto)}
        patcher = patch.object(release, 'protocol_sources', side_effect=lambda: self.sources)
        patcher.start()
        self.addCleanup(patcher.stop)
        rows, previous = [], []
        for model, batch in sorted(release.CELLS):
            files = {}
            for name in ('exported_program', 'checkpoint', 'manifest', 'inputs'):
                path = self.root / f'{model}_{batch}_{name}'
                path.write_text(f'{model}/{batch}/{name}')
                files[name] = dict(path=str(path), sha256=release.runner.sha(path))
            digest = f'inputs-{model}-{batch}'
            rows.append(dict(model=model, batch=batch, status='ir_created', precision='bf16',
                outputs=[dict(dtype='bfloat16')], asset_files=files, inputs_sha256=digest))
            if (model, batch) in release.RETAINED:
                previous.append(dict(model=model, batch=batch, system='aitemplate', status='pass',
                    precision='bf16', correctness_passes='5/5',
                    exported_program_sha256=files['exported_program']['sha256'],
                    source_checkpoint_sha256=files['checkpoint']['sha256'], inputs_sha256=digest))
        self.cpu = self.root / 'cpu.json'
        self.cpu_value = dict(identity=self.identity, rows=rows, input_seed=20261007,
            workload_id=release.WORKLOAD_ID, workload=release.WORKLOAD)
        self.cpu.write_text(json.dumps(self.cpu_value))
        retained = self.root / 'old10.csv'
        with retained.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(previous[0]))
            writer.writeheader()
            writer.writerows(previous)
        preparation = self.root / 'asset_preparation.json'
        prepared = [dict(model=row['model'], batch=row['batch'], status='ready',
            exported_program_sha256=row['asset_files']['exported_program']['sha256'],
            source_checkpoint_sha256=row['asset_files']['checkpoint']['sha256'],
            manifest_sha256=row['asset_files']['manifest']['sha256'],
            inputs_file_sha256=row['asset_files']['inputs']['sha256'],
            expected_inputs_sha256=row['inputs_sha256']) for row in rows
            if (row['model'], row['batch']) not in release.RETAINED]
        preparation.write_text(json.dumps(dict(CPU_assets_ready=True,
            active_models=list(release.MODELS), cells=prepared)))
        evidence = lambda p: dict(path=str(p), sha256=release.runner.sha(p))
        self.config = dict(workload_id=release.WORKLOAD_ID, workload=copy.deepcopy(release.WORKLOAD),
            parallel_baselines_authorized=True, gpu_dispatch_authorized=True,
            authorization='explicit source/asset freeze and parallel GPU dispatch',
            cpu_prepare=str(self.cpu), cpu_prepare_sha256=release.runner.sha(self.cpu), retained_AIT=evidence(retained),
            asset_preparation=evidence(preparation))
        self.destination = self.root / 'epoch'
        self.destination.mkdir()

    def test_complete_classic_release_has_no_TVM_or_primary_completion_dependency(self):
        publication = release.create(self.config, self.destination)
        self.assertEqual(publication['workload_id'], 'dnn-classic-v1')
        value = release.read(publication['path'])
        self.assertTrue(value['historical_TVM25_is_not_a_release_dependency'])
        self.assertEqual(len(value['assets']), 25)
        self.assertNotIn('tvm_epoch', value)

    def test_CPU_ready_does_not_authorize_GPU_or_old_workload(self):
        for key in ('parallel_baselines_authorized', 'gpu_dispatch_authorized'):
            config = copy.deepcopy(self.config)
            config[key] = False
            with self.assertRaisesRegex(ValueError, 'authorization'):
                release.create(config, self.destination)
        config = copy.deepcopy(self.config)
        config['workload']['models'][1] = 'mbv1'
        with self.assertRaisesRegex(ValueError, 'exact new classic25'):
            release.create(config, self.destination)

    def test_asset_or_protocol_drift_invalidates_release(self):
        publication = release.create(self.config, self.destination)
        path = Path(self.cpu_value['rows'][0]['asset_files']['inputs']['path'])
        path.write_text('changed-inputs')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            release.validate(publication['path'])

    def test_source_drift_and_missing_cells_are_rejected(self):
        self.cpu_value['rows'].pop()
        self.cpu.write_text(json.dumps(self.cpu_value))
        self.config['cpu_prepare_sha256'] = release.runner.sha(self.cpu)
        with self.assertRaisesRegex(ValueError, 'incomplete or changed'):
            release.create(self.config, self.destination)

    def test_reused_ten_require_the_original_asset_values(self):
        next(row for row in self.cpu_value['rows']
             if (row['model'], row['batch']) in release.RETAINED)['inputs_sha256'] = 'changed-digest'
        self.cpu.write_text(json.dumps(self.cpu_value))
        self.config['cpu_prepare_sha256'] = release.runner.sha(self.cpu)
        with self.assertRaisesRegex(ValueError, 'different assets'):
            release.create(self.config, self.destination)

    def test_existing_release_and_sample_input_changes_cannot_silently_resume(self):
        publication = release.create(self.config, self.destination)
        with self.assertRaises(FileExistsError):
            release.create(self.config, self.destination)
        asset = release.read(publication['path'])['assets'][0]
        args = [publication, asset['model'], asset['batch'],
                asset['files']['exported_program']['sha256'],
                asset['files']['checkpoint']['sha256'], asset['inputs_sha256']]
        self.assertEqual(release.sample_assets(*args), asset)
        args[-1] = 'different-input-values'
        with self.assertRaisesRegex(ValueError, 'sample differs'):
            release.sample_assets(*args)

    def test_CPU_proof_and_source_cannot_change_after_freeze(self):
        publication = release.create(self.config, self.destination)
        self.identity = dict(source='changed-source')
        with self.assertRaisesRegex(ValueError, 'source differs'):
            release.validate(publication['path'])

    def test_current_card_six_uses_old_seven_NUMA_affinity(self):
        self.assertEqual(cell.AFFINITY[6], list(range(64, 72)))


if __name__ == '__main__':
    unittest.main()
