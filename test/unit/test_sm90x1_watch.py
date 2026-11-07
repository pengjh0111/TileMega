"""CPU acceptance gates and local Git publication fixtures for the watcher."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import sm90x1_watch as watch


class AcceptanceTest(unittest.TestCase):
    def test_incomplete_receipts_do_not_allow_publication(self):
        value = dict(required_cells=14, timed_cells=14, trace_coordinates=165, total_traces=165)
        self.assertTrue(watch.accepted_counts(value))
        for field in ('timed_cells', 'trace_coordinates', 'total_traces'):
            incomplete = dict(value);incomplete[field] -= 1
            self.assertFalse(watch.accepted_counts(incomplete))

    def correctness(self):
        return dict(pass_=True, c1_pass=True, c2_pass=True,
                    spec=dict(kind='corr', correctness=dict(c1=256, c2=True, reference_processes=1)),
                    fresh_processes=5, processes_token_equal=5,
                    rows=[dict(c1=dict(pass_=True), c2=dict(pass_=True)) for _ in range(5)])

    def test_correctness_requires_the_actual_c1_and_c2_receipts(self):
        result = self.correctness();result['pass'] = result.pop('pass_')
        for row in result['rows']:
            for key in ('c1', 'c2'):row[key]['pass'] = row[key].pop('pass_')
        watch.check_correctness(result)
        for change in ('c1', 'c2', 'processes', 'protocol'):
            bad = copy.deepcopy(result)
            if change == 'c1':bad['rows'][0].pop('c1')
            elif change == 'c2':bad['rows'][3]['c2']['pass'] = False
            elif change == 'processes':bad['fresh_processes'] = 4
            else:bad['spec']['correctness']['c1'] = 64
            with self.subTest(change=change), self.assertRaises(ValueError):watch.check_correctness(bad)

    def timing(self, n):
        labels = ['W', 'L', 'Wstar']
        order = [label for i in range(n) for label in (labels if i % 2 == 0 else labels[::-1])]
        return dict(spec=dict(kind='sel', order=order), rows=[dict(arm=label) for label in order],
                    arms={label:dict(processes=n, tpot_ms=1., range_pct=.2) for label in labels})

    def test_three_and_five_fresh_process_timing_protocols(self):
        for n in (3, 5):watch.check_timing(self.timing(n), dict(processes=n))

    def test_schema_five_defaults_preserve_the_actual_structure(self):
        reference = dict(residency=1, cluster=1, emb='stage', group_wait=True)
        watch.check_structure(dict(reference, expert_fuse=False), reference)
        with self.assertRaises(ValueError):watch.check_structure(dict(reference, expert_fuse=True), reference)
        with self.assertRaises(ValueError):watch.check_structure(dict(reference, cluster=2), reference)

    def test_grouped_incomplete_and_nonfinite_timings_are_rejected(self):
        good = self.timing(5)
        for change in ('order', 'rows', 'count', 'nonfinite', 'arm'):
            bad = copy.deepcopy(good)
            if change == 'order':bad['spec']['order'].sort()
            elif change == 'rows':bad['rows'].pop()
            elif change == 'count':bad['arms']['W']['processes'] = 4
            elif change == 'nonfinite':bad['arms']['L']['tpot_ms'] = float('nan')
            else:bad['arms'].pop('Wstar')
            with self.subTest(change=change), self.assertRaises(ValueError):
                watch.check_timing(bad, dict(processes=5))

    def test_figures_require_correct_normalization_and_complete_scatter(self):
        arms = [dict(cell='cell', arm=arm, status='measured', tpot_ms=ms, gpu='0', **{'class':'trial'})
                for arm, ms in [('W', '12'), ('L', '11'), ('Wstar', '10')]]
        normalized = [dict(cell='cell', arm=arm, normalized_tpot=ratio, gpu='0', **{'class':'trial'})
                      for arm, ratio in [('W', '1.2'), ('L', '1.1'), ('Wstar', '1')]]
        candidates = [dict(cell='cell', family=f, variant='W', own_us='1', step_us='10',
                           processes='2', gpu='0', **{'class':'trial'}) for f in ('qkv', 'attention')]
        watch.check_figures(arms, candidates, normalized, copy.deepcopy(candidates))
        bad = copy.deepcopy(normalized);bad[0]['normalized_tpot'] = '1.3'
        with self.assertRaises(ValueError):watch.check_figures(arms, candidates, bad, candidates)
        bad = copy.deepcopy(candidates);bad[0]['own_us'] = '2'
        with self.assertRaises(ValueError):watch.check_figures(arms, candidates, normalized, bad)
        with self.assertRaises(ValueError):watch.check_figures(arms, candidates, normalized, candidates[:1])

    def test_live_campaign_once_does_not_finalize_or_push(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def read(path):
                return dict(status='running') if Path(path).name == 'campaign-state.json' else dict(pid=os.getpid())
            with patch.object(watch, 'RUN', root), patch.object(watch, 'STATE', root/'state.json'), \
                 patch.object(watch, 'check_setup', return_value={}), patch.object(watch, 'read', side_effect=read), \
                 patch.object(watch, 'progress', return_value=dict(timed_cells=12)), \
                 patch.object(watch, 'finish') as finish, patch.object(sys, 'argv', ['watch', '--once', '--push']), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(watch.main(), 2)
            finish.assert_not_called()
            self.assertEqual(json.loads((root/'state.json').read_text())['status'], 'watching')

    def test_terminal_but_incomplete_campaign_does_not_finalize(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(watch, 'RUN', root), patch.object(watch, 'STATE', root/'state.json'), \
                 patch.object(watch, 'check_setup', return_value={}), \
                 patch.object(watch, 'read', return_value=dict(status='complete')), \
                 patch.object(watch, 'progress', return_value=dict(required_cells=14, timed_cells=12,
                            trace_coordinates=165, total_traces=165)), \
                 patch.object(watch, 'finish') as finish, patch.object(sys, 'argv', ['watch', '--once']):
                with self.assertRaises(ValueError):watch.main()
            finish.assert_not_called()

    def test_five_fresh_processes_cannot_duplicate_the_monitor(self):
        code = '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from sm90x1_watch import monitor_lock
try:
    with monitor_lock(Path(sys.argv[2])): print('acquired')
except BlockingIOError: print('busy')
'''
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp)/'watch.lock'
            argv = [sys.executable, '-c', code, str(SCRIPTS), str(lock)]
            with watch.monitor_lock(lock):
                children = [subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='')) for _ in range(5)]
                for child in children:
                    out, err = child.communicate(timeout=30)
                    self.assertEqual(child.returncode, 0, err)
                    self.assertEqual(out.strip(), 'busy')
            self.assertEqual(subprocess.check_output(argv, text=True).strip(), 'acquired')


class GitTest(unittest.TestCase):
    def run_git(self, root, *argv):
        return subprocess.check_output(['git', *argv], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()

    def repository(self, temp):
        root = Path(temp);remote = root/'remote.git';local = root/'local';local.mkdir()
        self.run_git(root, 'init', '--bare', '-b', 'tilemega', str(remote))
        self.run_git(local, 'init', '-b', 'tilemega')
        self.run_git(local, 'config', 'user.email', 'fixture@example.invalid')
        self.run_git(local, 'config', 'user.name', 'CPU fixture')
        (local/'.gitignore').write_text('runs/\n')
        (local/'include').mkdir();(local/'include/frozen.cc').write_text('frozen\n')
        (local/'scripts').mkdir();(local/'scripts/tool.py').write_text('# fixture\n')
        self.run_git(local, 'add', '.');self.run_git(local, 'commit', '-m', 'test: create fixture')
        self.run_git(local, 'remote', 'add', 'origin', str(remote))
        self.run_git(local, 'push', 'origin', 'HEAD:tilemega')
        run = local/'runs/sm90x1';run.mkdir(parents=True)
        manifest = dict(measurement_head=self.run_git(local, 'rev-parse', 'HEAD'),
                        files={'scripts/tool.py':hashlib.sha256((local/'scripts/tool.py').read_bytes()).hexdigest()},
                        commit_groups=[])
        (run/'pending-build-tooling.json').write_text(json.dumps(manifest))
        return local, remote, run

    def test_setup_rejects_frozen_source_and_unexpected_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            root, remote, run = self.repository(temp)
            with patch.object(watch, 'ROOT', root), patch.object(watch, 'RUN', run), \
                 patch.object(watch, 'cells', return_value=[{}]*14):
                watch.check_setup()
                source = root/'include/frozen.cc';source.write_text('changed\n')
                with self.assertRaises(ValueError):watch.check_setup()
                source.write_text('frozen\n');(root/'unexpected.txt').write_text('unrelated\n')
                with self.assertRaises(ValueError):watch.check_setup()

    def test_fast_forward_push_is_confirmed_against_remote(self):
        with tempfile.TemporaryDirectory() as temp:
            root, remote, run = self.repository(temp)
            (root/'result.csv').write_text('metric,value\nX,1.01\n')
            self.run_git(root, 'add', '.');self.run_git(root, 'commit', '-m', 'docs: record fixture result')
            expected = self.run_git(root, 'rev-parse', 'HEAD')
            with patch.object(watch, 'ROOT', root):proof = watch.push('origin', 'tilemega')
            self.assertEqual(proof['confirmed_remote_head'], expected)
            self.assertEqual(self.run_git(remote, 'rev-parse', 'tilemega'), expected)

    def test_dirty_workspace_is_not_pushed(self):
        with tempfile.TemporaryDirectory() as temp:
            root, remote, run = self.repository(temp);before = self.run_git(remote, 'rev-parse', 'tilemega')
            (root/'uncommitted.txt').write_text('pending\n')
            with patch.object(watch, 'ROOT', root), self.assertRaises(ValueError):watch.push('origin', 'tilemega')
            self.assertEqual(self.run_git(remote, 'rev-parse', 'tilemega'), before)

    def test_divergent_remote_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root, remote, run = self.repository(temp)
            (root/'other.txt').write_text('other developer\n');self.run_git(root, 'add', '.')
            self.run_git(root, 'commit', '-m', 'test: create remote successor')
            self.run_git(root, 'push', 'origin', 'HEAD:tilemega');remote_head = self.run_git(root, 'rev-parse', 'HEAD')
            self.run_git(root, 'checkout', '-b', 'local-result', 'HEAD~1')
            (root/'result.csv').write_text('X,1.02\n');self.run_git(root, 'add', '.')
            self.run_git(root, 'commit', '-m', 'docs: record local fixture')
            self.run_git(root, 'branch', '-D', 'tilemega');self.run_git(root, 'branch', '-m', 'tilemega')
            with patch.object(watch, 'ROOT', root), self.assertRaises(subprocess.CalledProcessError):
                watch.push('origin', 'tilemega')
            self.assertEqual(self.run_git(remote, 'rev-parse', 'tilemega'), remote_head)


if __name__ == '__main__':
    unittest.main()
