"""Only completed infrastructure failures are automatically requeued (CPU-only)."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('controller', ROOT / 'scripts/dnn1_tvm_collect_controller.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InfrastructureRequeue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, epoch='epoch', *, error='whole cell invalidated by timing guard/throttle',
                ended=10, status='hard_stop', code=0, live=False):
        root = self.root / epoch
        worker = root / 'dispatch/workers/resnet18_b16/run'
        session = worker / 'raw/dnn1-cell-resnet18_b16/attempt1/job/cell-result.json'
        session.parent.mkdir(parents=True)
        session.write_text(json.dumps(dict(model='resnet18', batch=16, status=status,
            error=error, ended=ended, logical_jobs=[dict(arm=dict(system='tvm', precision='bf16'),
                attempts=[dict(returncode=code)])])))
        state = dict(failures=[dict(root=str(worker), exit_code=1)],
            workers={'7': dict(run_root=str(worker))} if live else {})
        (root / 'dispatch/state.json').write_text(json.dumps(state))
        return root, session

    def test_guard_rejection_preserves_original_evidence_and_completed_cache(self):
        root, session = self.fixture()
        before = session.read_bytes()
        result = module.infrastructure_retries([root], set())
        row = result[('resnet18', 16)]
        self.assertTrue(row['completed_bf16_process'])
        self.assertEqual(row['session_sha256'], module.sha(session))
        self.assertEqual(session.read_bytes(), before)
        self.assertIn('other GPU may retune', row['cache_policy'])

    def test_partial_cache_requires_fresh_isolated_root(self):
        root, _ = self.fixture(error='TVM infrastructure rejection; no FP16 alternative: incomplete cache', code=1)
        row = module.infrastructure_retries([root], set())[('resnet18', 16)]
        self.assertFalse(row['completed_bf16_process'])
        self.assertEqual(row['cache_policy'], 'fresh isolated recovery root')

    def test_live_unfinished_and_numerical_failures_are_not_retried(self):
        for index, kw in enumerate([dict(live=True), dict(ended=None),
            dict(error='correctness cosine below original threshold'), dict(status='running')]):
            root, _ = self.fixture(str(index), **kw)
            self.assertEqual(module.infrastructure_retries([root], set()), {})

    def test_accepted_cell_or_later_genuine_failure_supersedes_old_rejection(self):
        root, _ = self.fixture()
        clean = {('resnet18', '16', 'tvm', 'tvm-bf16', 'bf16')}
        self.assertEqual(module.infrastructure_retries([root], clean), {})
        later, _ = self.fixture('later', error='genuine compiler error', ended=20)
        self.assertEqual(module.infrastructure_retries([root, later], set()), {})

    def test_later_incomplete_cache_does_not_hide_original_compile_failure(self):
        root, session = self.fixture(
            error='TVM infrastructure rejection; no FP16 alternative: incomplete cache', code=1)
        result = json.loads(session.read_text())
        result['logical_jobs'].insert(0, dict(arm=dict(system='tvm', precision='bf16'),
            attempts=[dict(returncode=1, failure=dict(failure_kind='compile',
                error='There are some consumers of the cache-write stage not inlined'))]))
        session.write_text(json.dumps(result))
        self.assertEqual(module.infrastructure_retries([root], set()), {})


if __name__ == '__main__':
    unittest.main()
