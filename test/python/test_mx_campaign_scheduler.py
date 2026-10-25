"""CPU-only campaign scheduling, immutable resume, and fail-fast coverage."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch


source = Path(__file__).resolve().parents[2] / 'experiments/mx5/schedule_campaign.py'
spec = importlib.util.spec_from_file_location('mx5_schedule_campaign', source)
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


class CampaignScheduler(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.jobs = []
        for index, kind in enumerate(('head', 'attention', 'head', 'gemm', 'head')):
            directory = self.root / f'job{index}'
            directory.mkdir()
            keys = directory / 'keys.json'
            keys.write_text(json.dumps([dict(shape=index, library='/fake/binary.so')]))
            self.jobs.append(dict(directory=str(directory), keys=str(keys), rows=1,
                                  kind=kind, keys_sha256=hashlib.sha256(keys.read_bytes()).hexdigest()))
        self.manifest = self.root / 'campaign.json'
        self.manifest.write_text(json.dumps(dict(processes_per_key=3, jobs=self.jobs)))
        self.args = argparse.Namespace(manifest=self.manifest, out=self.root / 'scheduler',
                                       jobs=None, start=0, stop=None, kinds=None, iterations=200)
        self.events = []
        self.launched = []
        self.active = set()
        self.max_active = 0

    def complete(self, index):
        job = self.jobs[index]
        directory = Path(job['directory'])
        keys = json.loads(Path(job['keys']).read_text())
        for round_number in (1, 2, 3):
            output = directory / f'round{round_number}.jsonl'
            output.write_text(''.join(json.dumps(dict(key=key, round=round_number)) + '\n' for key in keys))
            output.with_suffix('.log').write_text('successful fixture\n')
            process = dict(round=round_number, returncode=0,
                           command=['python', 'bench.py', '--keys', job['keys'],
                                    '--out', str(output), '--round', str(round_number)])
            (directory / f'process{round_number}.json').write_text(json.dumps(process))
        guard = directory / 'guard'
        guard.mkdir(exist_ok=True)
        (guard / 'guard_result.json').write_text(json.dumps(dict(code=0)))

    def fake_popen(self, sequences=None, spawn_error=None):
        sequences = sequences or {}
        outer = self

        def popen(command, **kwargs):
            index = int(command[command.index('--job') + 1])
            card = int(command[1])
            if spawn_error == index:
                raise OSError('fixture spawn failure')
            outer.assertNotIn(card, outer.active)
            outer.active.add(card)
            outer.max_active = max(outer.max_active, len(outer.active))
            outer.launched.append((index, card, command, kwargs))
            polls = iter(sequences.get(index, [None, 0]))

            class Process:
                def poll(self):
                    code = next(polls)
                    if code is not None:
                        outer.active.remove(card)
                        if code == 0:
                            outer.complete(index)
                    return code

            return Process()

        return popen

    def execute(self, fake):
        self.args.out.mkdir()
        selected = scheduler.select_jobs(json.loads(self.manifest.read_text()), self.args)
        with patch.object(scheduler.subprocess, 'Popen', side_effect=fake), \
                patch.object(scheduler.time, 'sleep'):
            return scheduler.execute(self.args, selected, scheduler.digest(self.manifest),
                                     lambda event, **data: self.events.append(dict(event=event, **data)))

    def test_two_cards_reuse_first_free_and_record_guard_command(self):
        self.assertEqual(self.execute(self.fake_popen({0: [None, 0], 1: [None, None, 0]})), 0)
        self.assertEqual([(i, card) for i, card, _, _ in self.launched][:3], [(0, 7), (1, 6), (2, 7)])
        self.assertEqual(self.max_active, 2)
        self.assertFalse(self.active)
        self.assertEqual(len([row for row in self.events if row['event'] == 'finish']), 5)
        for index, card, command, kwargs in self.launched:
            self.assertEqual(command[:4], [str(scheduler.ROOT / 'scripts/mx_gpu_job.sh'),
                                          str(card), '900', self.jobs[index]['directory'] + '/guard'])
            self.assertEqual(command[5:7], [str(scheduler.ROOT / 'experiments/mx5/campaign.py'), 'run'])
            self.assertEqual(kwargs['env']['NEEDS_MIB'], '20480')
            self.assertEqual(kwargs['env']['TILEMEGA_GPU_JOB_LOG'],
                             str(scheduler.ROOT / 'runs/mx5/gpu_jobs.log'))
            self.assertTrue(kwargs['start_new_session'])
            marker = json.loads((Path(self.jobs[index]['directory']) / 'scheduler_job.json').read_text())
            self.assertEqual(marker['command'], command)

    def test_failure_stops_launches_and_drains_other_guard(self):
        code = self.execute(self.fake_popen({0: [1], 1: [None, None, 0]}))
        self.assertEqual(code, 1)
        self.assertEqual([(i, card) for i, card, _, _ in self.launched], [(0, 7), (1, 6)])
        self.assertFalse(self.active)
        self.assertEqual([row['status'] for row in self.events if row['event'] == 'finish'],
                         ['failed', 'completed'])
        self.assertEqual([row['job'] for row in self.events if row['event'] == 'not_launched'], [2, 3, 4])

    def test_spawn_failure_drains_other_guard(self):
        self.assertEqual(self.execute(self.fake_popen(spawn_error=1)), 1)
        self.assertEqual([row[0] for row in self.launched], [0])
        self.assertFalse(self.active)
        self.assertEqual(scheduler.batch_state(self.jobs[0]), 'completed')
        with self.assertRaisesRegex(ValueError, 'partial or unsuccessful'):
            scheduler.batch_state(self.jobs[1])

    def test_second_card_failure_prevents_reusing_first_card(self):
        self.assertEqual(self.execute(self.fake_popen({0: [0], 1: [75]})), 1)
        self.assertEqual([row[0] for row in self.launched], [0, 1])
        self.assertFalse(self.active)

    def test_signal_stops_launches_and_drains_started_guard(self):
        fake = self.fake_popen()

        def interrupted_spawn(command, **kwargs):
            process = fake(command, **kwargs)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return process

        self.assertEqual(self.execute(interrupted_spawn), 128 + signal.SIGTERM)
        self.assertEqual([row[0] for row in self.launched], [0])
        self.assertFalse(self.active)
        self.assertEqual(scheduler.batch_state(self.jobs[0]), 'completed')

    def test_resume_skips_complete_matching_guarded_rounds(self):
        self.complete(0)
        before = {path: path.read_bytes() for path in Path(self.jobs[0]['directory']).rglob('*') if path.is_file()}
        self.assertEqual(self.execute(self.fake_popen()), 0)
        self.assertEqual([row[0] for row in self.launched], [1, 2, 3, 4])
        self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))

    def test_preflight_rejects_partial_before_any_launch(self):
        path = Path(self.jobs[2]['directory']) / 'round1.log'
        path.write_text('preserve me')
        with self.assertRaisesRegex(ValueError, 'partial or unsuccessful'):
            self.execute(self.fake_popen())
        self.assertEqual(path.read_text(), 'preserve me')
        self.assertFalse(self.launched)

    def test_resume_rejects_wrong_keys_rounds_process_and_guard(self):
        self.complete(0)
        directory = Path(self.jobs[0]['directory'])
        mutations = [
            ('round2.jsonl', json.dumps(dict(key=dict(shape='wrong'), round=2)) + '\n'),
            ('round2.jsonl', ''),
            ('round2.jsonl', json.dumps(dict(key=json.loads(Path(self.jobs[0]['keys']).read_text())[0], round=1))),
            ('process2.json', json.dumps(dict(round=2, returncode=1, command=[]))),
            ('guard/guard_result.json', json.dumps(dict(code=75))),
        ]
        for filename, replacement in mutations:
            with self.subTest(filename=filename, replacement=replacement):
                path = directory / filename
                original = path.read_text()
                path.write_text(replacement)
                with self.assertRaisesRegex(ValueError, 'partial or unsuccessful'):
                    scheduler.batch_state(self.jobs[0])
                path.write_text(original)
        Path(self.jobs[0]['keys']).write_text('[]')
        with self.assertRaisesRegex(ValueError, 'campaign keys changed'):
            scheduler.batch_state(self.jobs[0])

    def test_selection_intersects_bounds_indices_and_kinds(self):
        self.args.start, self.args.stop = 1, 5
        self.args.jobs = [0, 1, 2, 4]
        self.args.kinds = ['head']
        selected = scheduler.select_jobs(dict(jobs=self.jobs), self.args)
        self.assertEqual([index for index, _ in selected], [2, 4])
        self.args.jobs = [5]
        with self.assertRaisesRegex(ValueError, 'index outside'):
            scheduler.select_jobs(dict(jobs=self.jobs), self.args)

    def test_exclusive_output_and_campaign_lock(self):
        self.args.out.mkdir()
        with self.assertRaises(FileExistsError):
            scheduler.schedule(self.args)
        self.assertFalse(self.launched)
        with patch.object(scheduler.fcntl, 'flock', side_effect=BlockingIOError):
            with self.assertRaisesRegex(ValueError, 'another scheduler'):
                scheduler.schedule(self.args)

    def test_preflight_failure_is_preserved_in_exclusive_record(self):
        (Path(self.jobs[0]['directory']) / 'round1.log').write_text('partial')
        with contextlib.redirect_stdout(io.StringIO()), patch.object(scheduler.subprocess, 'Popen') as popen:
            self.assertEqual(scheduler.schedule(self.args), 1)
        popen.assert_not_called()
        self.assertEqual(json.loads((self.args.out / 'result.json').read_text()), dict(returncode=1))
        self.assertEqual(json.loads((self.args.out / 'scheduler.json').read_text())['selected'], list(range(5)))
        rows = [json.loads(line) for line in (self.args.out / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['event'], 'error')


if __name__ == '__main__':
    unittest.main()
