#!/usr/bin/env python3
"""CPU-only checks of the dnn3-v1 parameterization; synthetic files are never evidence."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn1_tvm_trials10000_collect as collect
import dnn1_tvm_trials10000_sample as sample
import dnn3_make_queues as make

base = collect.base
ROOT = collect.ROOT
WORKLOAD = json.loads((ROOT / 'configs/dnn/dnn3_workload.json').read_text())
DEVICES = ROOT / 'configs/dnn/dnn3_h800_devices.json'
TUNE = dict(id='dnn3-v1', models=['resnet50', 'vgg16'], batches=[1, 2, 4, 8, 16])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def exports(self, models=('resnet50', 'vgg16')):
        root = self.root / 'exports'
        for model in models:
            export = root / model
            (export / 'checkpoint').mkdir(parents=True)
            (export / 'exported_program.pt2').write_bytes(model.encode())
            (export / 'checkpoint/model.safetensors').write_bytes(b'weights ' + model.encode())
            sample.save(export / 'manifest.json', dict(model=model, timing_eligible=True,
                artifacts={'exported_program.pt2': sample.sha(export / 'exported_program.pt2')},
                checkpoint=dict(sha256=sample.sha(export / 'checkpoint/model.safetensors'))))
            for batch in sample.BATCHES:
                (export / f'inputs_b{batch}.safetensors').write_bytes(b'inputs%d' % batch)
        return root


class QueueWorkload(Base):
    def test_classic_defaults_are_the_module_constants(self):
        value = base.queue_workload({})
        self.assertEqual(value['id'], 'dnn-classic-v1')
        self.assertEqual(value['models'], base.MODELS)
        self.assertEqual(value['batches'], base.BATCHES)
        self.assertIsNone(value['arms'])
        self.assertEqual(value['device_file'], base.CLASSIC_DEVICES)
        self.assertEqual(base.queue_workload(dict(workload_id='dnn-classic-v1'))['models'], base.MODELS)

    def test_classic_refuses_overrides_and_unknown_ids_stay_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no queue-spec overrides'):
            base.queue_workload(dict(models=['resnet50']))
        with self.assertRaisesRegex(ValueError, 'active classic workload'):
            base.queue_workload(dict(workload_id='historical-workload'))

    def test_spec_workload_must_declare_everything_and_exclude_fp16(self):
        spec = {key: WORKLOAD[key] for key in ('workload_id', 'models', 'batches', 'arms', 'device_file')}
        value = base.queue_workload(spec)
        self.assertEqual(value['models'], ('resnet50', 'vgg16'))
        self.assertEqual(value['active_gpus'], (0, 1, 2, 3))
        self.assertNotIn('tensorrt-fp16', value['arms'])
        for name in ('models', 'batches', 'arms', 'device_file'):
            with self.assertRaisesRegex(ValueError, 'must declare'):
                base.queue_workload({k: v for k, v in spec.items() if k != name})
        with self.assertRaisesRegex(ValueError, 'BF16 baselines only'):
            base.queue_workload(dict(spec, arms=WORKLOAD['arms'] + ['tensorrt-fp16']))
        with self.assertRaisesRegex(ValueError, 'invalid spec workload'):
            base.queue_workload(dict(spec, arms=['tvm-bf16']))
        with self.assertRaisesRegex(ValueError, 'another workload'):
            base.queue_workload(dict(spec, device_file=str(base.CLASSIC_DEVICES)))

    def test_device_file_matches_the_classic_identities_the_guard_provider_uses(self):
        classic = json.loads(base.CLASSIC_DEVICES.read_text())['devices']
        devices = json.loads(DEVICES.read_text())
        self.assertEqual(devices['schema'], 1)
        self.assertEqual(devices['allowed_gpu_indices'], [0, 1, 2, 3])
        for gpu in range(4):
            self.assertEqual(devices['devices'][str(gpu)], classic[str(gpu)])
            self.assertEqual(devices['devices'][str(gpu)]['numa_node'], 0)
        value = dict(device_file=DEVICES)
        self.assertEqual(base.workload_device_identity(2, value)['pci'], '0000:41:00.0')
        with self.assertRaisesRegex(ValueError, 'allowed indices'):
            base.workload_device_identity(4, value)

    def build(self, gpu=2, model='resnet50', **overrides):
        exports = self.exports()
        spec = make.spec_for(WORKLOAD, gpu, model, exports)
        spec.update(python=sys.executable, inductor_cache=str(self.root / 'inductor'), **overrides)
        path = self.root / 'input.json'
        sample.save(path, spec)
        out = self.root / 'run'
        with patch('tilemega.bench.gpu_guard.require_allowed_gpu'), redirect_stdout(io.StringIO()):
            base.queue(SimpleNamespace(spec=path, out=out, refresh_prepared=False))
        return out

    def test_queue_has_four_arms_and_binds_card_cores_and_device_file(self):
        out = self.build()
        queued = sample.read(out / 'queue-spec.json')
        self.assertEqual(queued['workload_id'], 'dnn3-v1')
        self.assertEqual(queued['workload'], dict(models=['resnet50', 'vgg16'], batches=[1, 2, 4, 8, 16]))
        self.assertEqual(len(queued['cells']), 5)
        cell = sample.read(queued['cells'][0])
        self.assertEqual([arm['arm'] for arm in cell['arms']], WORKLOAD['arms'])
        self.assertEqual(cell['order'][1], list(reversed(WORKLOAD['arms'])))
        self.assertEqual(cell['protocol'], base.PROTOCOL)
        self.assertEqual(cell['workload_id'], 'dnn3-v1')
        self.assertEqual(cell['active_gpus'], [0, 1, 2, 3])
        self.assertEqual(cell['correctness_policy'], 'five-independent-gates-v2')
        self.assertEqual(cell['cpu_affinity'], list(range(16, 24)))
        self.assertEqual(cell['inductor_cache'], str(self.root / 'inductor/gpu2'))
        self.assertEqual(cell['recipe_sources'][str(DEVICES)], sample.sha(DEVICES))
        # dnn1_device still reads the classic map, so it is frozen with the queue.
        self.assertEqual(cell['recipe_sources'][str(base.CLASSIC_DEVICES)], sample.sha(base.CLASSIC_DEVICES))
        command = sample.read(out / 'queue/queue_dnn1_baselines.json')[1]['command']
        self.assertIn('--physcpubind=' + ','.join(map(str, range(16, 24))), command)
        self.assertEqual(command[command.index('--expected-pci') + 1], '0000:41:00.0')
        self.assertEqual(command[command.index('--numa-node') + 1], '0')

    def test_spec_workload_refuses_the_dnn1_asset_root_fallback(self):
        spec = make.spec_for(WORKLOAD, 2, 'resnet50', self.root)
        del spec['inductor_cache']
        path = self.root / 'input.json'
        sample.save(path, spec)
        with self.assertRaisesRegex(ValueError, 'must not fall back'):
            base.queue(SimpleNamespace(spec=path, out=self.root / 'run', refresh_prepared=False))

    def test_preflight_is_ready_for_an_unchanged_queue(self):
        out = self.build()
        with redirect_stdout(io.StringIO()):
            base.preflight(SimpleNamespace(run_root=out))
        self.assertEqual(sample.read(out / 'preflight.json')['status'], 'ready')

    def test_require_session_rejects_a_card_outside_the_workload(self):
        with self.assertRaisesRegex(base.HardStop, 'declared device indices'):
            base.require_session(dict(gpu=5, workload_id='dnn3-v1', active_gpus=[0, 1, 2, 3]))

    def test_collect_names_the_workload_and_adds_its_column(self):
        run = self.root / 'collected-run'
        sample.save(run / 'queue-spec.json', dict(workload_id='dnn3-v1', cells=[],
                                                  config=dict(output_stem='DNN3')))
        sample.save(run / 'scheduler/state.json', {})
        with redirect_stdout(io.StringIO()):
            base.collect(SimpleNamespace(run_root=run, out=self.root / 'out'))
        header = (self.root / 'out/DNN3_baselines.csv').read_text().splitlines()[0].split(',')
        self.assertEqual(header, base.FIELDS + ['workload_id'])
        self.assertFalse((self.root / 'out/DNN1_baselines.csv').exists())


class TvmWorkload(Base):
    def manifest(self, epoch, **overrides):
        value = dict(profile=sample.PROFILE, budget=sample.BUDGET, fresh_database=True, official_cuda_graph=False,
            workload_id='dnn3-v1', workload=TUNE, release_gate='not_applicable_dnn3',
            allowed_gpu_indices=[0, 1, 2, 3], epoch=str(epoch),
            asset_preparation=dict(path=str(self.root / 'proof.json'), sha256=''),
            cells=[dict(model=m, batch=b) for m in TUNE['models'] for b in TUNE['batches']],
            max_parallel=2, other_baseline_max_parallel=0, process_timeout_s=18000, cell_timeout_s=108000,
            gpus=[2, 3], protocol_sources={}, protected_files={}, tvm_root=str(self.root / 'tvm'),
            source_identity=dict(source_diff_sha256=hashlib.sha256(b'').hexdigest()))
        value.update(overrides)
        sample.save(self.root / 'proof.json', dict(synthetic=True))
        value['asset_preparation']['sha256'] = sample.sha(self.root / 'proof.json')
        sample.save(epoch / 'manifest.json', value)
        return value

    def test_manifest_records_gate_and_needs_no_release(self):
        epoch = self.root / 'runs/dnn3/tvm_fresh10000_v1'
        self.manifest(epoch)
        with patch.object(collect, 'ROOT', self.root), patch.object(collect.subprocess, 'check_output', return_value=b''):
            manifest, digest = collect.validate_release(epoch)
            self.assertIsNone(digest)
            self.assertEqual(manifest['release_gate'], 'not_applicable_dnn3')
            with self.assertRaisesRegex(ValueError, 'no AIT release to seal'):
                collect.seal_release(SimpleNamespace(epoch=epoch))
            sample.save(epoch / 'release.json', dict(synthetic=True))
            with self.assertRaisesRegex(ValueError, 'has no AIT release'):
                collect.validate_release(epoch)

    def test_classic_cannot_shed_the_release_or_leave_its_root(self):
        self.assertTrue(sample.release_required(dict(workload_id=sample.WORKLOAD_ID, release_gate='not_applicable_dnn3')))
        self.assertTrue(sample.release_required({}))
        with self.assertRaisesRegex(ValueError, 'unknown workload or release gate'):
            sample.release_required(dict(workload_id='dnn3-v1', release_gate='anything'))
        with self.assertRaisesRegex(ValueError, 'unknown workload or release gate'):
            sample.release_required(dict(workload_id='other', release_gate='not_applicable_dnn3'))
        epoch = self.root / 'runs/dnn1/wrong_root'
        self.manifest(epoch)
        with patch.object(collect, 'ROOT', self.root), patch.object(collect.subprocess, 'check_output', return_value=b''), \
             self.assertRaisesRegex(ValueError, 'epoch binding differs'):
            collect.validate_manifest(epoch)

    def test_wrong_cells_or_gate_rejected(self):
        epoch = self.root / 'runs/dnn3/epoch'
        self.manifest(epoch, cells=[dict(model='resnet50', batch=1)])
        with patch.object(collect, 'ROOT', self.root), patch.object(collect.subprocess, 'check_output', return_value=b''), \
             self.assertRaisesRegex(ValueError, 'distinct cells'):
            collect.validate_manifest(epoch)
        self.manifest(epoch, release_gate='ait')
        with patch.object(collect, 'ROOT', self.root), patch.object(collect.subprocess, 'check_output', return_value=b''), \
             self.assertRaisesRegex(ValueError, 'release gate'):
            collect.validate_manifest(epoch)

    def test_gpu_pool_follows_the_workload_device_file(self):
        self.assertEqual(collect.gpu_pool('2,3', [0, 1, 2, 3]), [2, 3])
        with self.assertRaises(ValueError):
            collect.gpu_pool('4', [0, 1, 2, 3])

    def test_asset_proof_covers_exactly_the_two_models(self):
        exports = self.exports()
        cells = []
        for model in ('resnet50', 'vgg16'):
            export = (exports / model).resolve()
            for batch in sample.BATCHES:
                inputs = export / f'inputs_b{batch}.safetensors'
                cells.append(dict(model=model, batch=batch, status='ready', export=str(export),
                    exported_program_sha256=sample.sha(export / 'exported_program.pt2'),
                    source_checkpoint_sha256=sample.sha(export / 'checkpoint/model.safetensors'),
                    manifest_sha256=sample.sha(export / 'manifest.json'), inputs_file=str(inputs),
                    inputs_file_sha256=sample.sha(inputs), expected_inputs_sha256='fixed'))
        proof = dict(schema=1, status='ready', seed=20261007, models=['resnet50', 'vgg16'], cells=cells,
                     identity_source='preparation_dnn3_v1 CPU runtime-input proof')
        rows = collect.asset_cells(exports, proof, [], ('resnet50', 'vgg16'), (), sample.BATCHES)
        self.assertEqual([(r['model'], r['batch']) for r in rows][:6],
                         [('resnet50', b) for b in sample.BATCHES] + [('vgg16', 1)])
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]['identity_source'], 'preparation_dnn3_v1 CPU runtime-input proof')
        with self.assertRaisesRegex(ValueError, 'exactly10'):
            collect.asset_cells(exports, dict(proof, cells=cells[:-1]), [], ('resnet50', 'vgg16'), (), sample.BATCHES)

    def test_tuning_cores_are_disjoint_numa1_and_cover_the_pool(self):
        cpus = set(range(192))
        with patch.object(collect.os, 'sched_getaffinity', return_value=cpus):
            value = collect.tuning_affinities(WORKLOAD, [2, 3])
        self.assertEqual(value, {'2': list(range(48, 72)), '3': list(range(72, 96))})
        with patch.object(collect.os, 'sched_getaffinity', return_value=cpus):
            with self.assertRaisesRegex(ValueError, 'declared tuner cores'):
                collect.tuning_affinities(WORKLOAD, [0, 2])
            for bad in (dict(WORKLOAD['tuning_cpu_affinity'], **{'3': list(range(60, 72))}),
                        dict(WORKLOAD['tuning_cpu_affinity'], **{'3': [16, 17]})):
                with self.assertRaisesRegex(ValueError, 'tuner cores must'):
                    collect.tuning_affinities(dict(WORKLOAD, tuning_cpu_affinity=bad), [2, 3])

    def test_real_child_inherits_tuner_mask_and_parent_restores(self):
        allowed = sorted(os.sched_getaffinity(0))
        before = set(allowed)
        tuner, keep = allowed[:2], allowed[2:]
        directory = self.root / 'cache/tune'
        def original(*args):
            directory.mkdir(parents=True)
            sample.save(directory / 'request.json', dict(max_trials_global=args[4]))
            child = subprocess.check_output([sys.executable, '-c',
                'import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'], text=True)
            result = dict(maximum_trials=args[4], timeout_s=args[5], child_cpus=json.loads(child))
            sample.save(directory / 'receipt.json', result)
            return 'database', result
        cell = dict(batch=1, model='resnet50', cache=str(directory.parent), tuning_cpu_affinity=tuner)
        _, result = sample.tuning_callback(original, cell)('fixed.pt2', 'bf16', 1, directory)
        self.assertEqual(result['child_cpus'], tuner)
        self.assertEqual(result['tuning_cpu_affinity'], tuner)
        self.assertEqual(os.sched_getaffinity(0), before)
        self.assertTrue(keep)

    def test_tuner_mask_restored_after_a_failed_tune_and_classic_untouched(self):
        before = os.sched_getaffinity(0)
        directory = self.root / 'cache/tune'
        cell = dict(batch=1, model='resnet50', cache=str(directory.parent),
                    tuning_cpu_affinity=sorted(before)[:1])
        with self.assertRaises(RuntimeError):
            sample.tuning_callback(Mock(side_effect=RuntimeError('tuner failed')), cell)('fixed.pt2', 'bf16', 1, directory)
        self.assertEqual(os.sched_getaffinity(0), before)
        with patch.object(sample.os, 'sched_setaffinity') as change, sample.tuning_affinity(dict(batch=1)):
            pass
        change.assert_not_called()

    def test_recipe_identity_and_cell_validation_use_the_declared_workload(self):
        recipe = sample.recipe(dict(schema=6), dict(workload_id='dnn3-v1', workload=TUNE, epoch_manifest_sha256='e'))
        self.assertEqual(recipe['workload'], TUNE)
        self.assertEqual(recipe['schema'], sample.PROFILE)
        self.assertEqual(recipe['tuning'], sample.BUDGET)
        self.assertEqual(sample.BUDGET['max_trials_global'], 10000)
        self.assertEqual(collect.PROCESS_TIMEOUT, 18000)
        self.assertEqual(sample.recipe({}, dict(epoch_manifest_sha256='e'))['workload'], sample.WORKLOAD)
        sample.validate_workload(dict(workload_id='dnn3-v1', workload=TUNE, model='resnet50', batch=16))
        with self.assertRaisesRegex(ValueError, 'outside its declared'):
            sample.validate_workload(dict(workload_id='dnn3-v1', workload=TUNE, model='resnet18', batch=1))
        with self.assertRaisesRegex(ValueError, 'active classic five-model'):
            sample.validate_workload(dict(workload_id=sample.WORKLOAD_ID, workload=sample.WORKLOAD, model='resnet50', batch=1))
        self.assertEqual(list(sample.allowed_gpus(dict(workload_id='dnn3-v1', allowed_gpu_indices=[0, 1, 2, 3]))), [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
