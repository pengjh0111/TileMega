import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = load('cleanup_inventory')
cleanup = load('cleanup_plan')


def row(path, category='A', owner='developer', size=16):
    return dict(path=str(path), category=category, owner=owner, bytes=size, action='review')


class CleanupPlanTest(unittest.TestCase):
    def test_detached_unpublished_tip_ignored_source_and_edits_are_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def git(*args):
                return subprocess.check_output(['git', '-C', folder, *args], text=True).strip()
            git('init', '-q')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.invalid')
            (root / '.gitignore').write_text('build/\n')
            (root / 'tracked.py').write_text('base\n')
            git('add', '.')
            git('commit', '-qm', 'base')
            base = git('rev-parse', 'HEAD')
            git('update-ref', 'refs/remotes/origin/main', base)
            git('checkout', '-q', '--detach')
            (root / 'tracked.py').write_text('detached\n')
            git('commit', '-qam', 'detached unpublished tip')
            head = git('rev-parse', 'HEAD')
            (root / 'tracked.py').write_text('dirty\n')
            (root / 'local.mlir').write_text('module {}\n')
            (root / 'build').mkdir()
            (root / 'build/custom_kernel.cu').write_text('// local source\n')
            status = inventory.git(root, 'status', '--porcelain=v1', '--ignored', '-z')
            audit = inventory.repository_audit(root, status)
            self.assertEqual(audit['head'], head)
            self.assertTrue(any(line.startswith(head) for line in audit['unpushed']))
            self.assertEqual(audit['untracked_source'], [str(root / 'local.mlir')])
            self.assertEqual(audit['ignored_source'], [str(root / 'build/custom_kernel.cu')])
            self.assertEqual(audit['tracked_changes'][0]['path'], str(root / 'tracked.py'))

    def test_dry_run_protects_data_and_code_and_deduplicates_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build, data, work = root / 'build-test', root / 'models', root / 'worktree'
            rows = [row(root, 'D'), row(root / '.git', 'F'), row(build, 'B', size=32),
                    row(build / 'kernel.so', 'A'), row(data, 'E'), row(data / 'large.bin', 'F'),
                    row(work, 'D'), row(root / 'other-user', owner='someone-else')]
            audits = [dict(root=str(root), tracked_changes=[], unpushed=[], stash=[]),
                      dict(root=str(work), tracked_changes=[], unpushed=['local tip'], stash=[])]
            result = cleanup.plan(dict(repository=str(root), worktrees=audits), rows, owner='developer')
            by_path = {r['path']: r for r in result}
            self.assertEqual(by_path[str(root)]['action'], 'blocked')
            self.assertEqual(by_path[str(build)]['action'], 'candidate')
            self.assertEqual(by_path[str(build / 'kernel.so')]['action'], 'covered')
            self.assertEqual(by_path[str(data)]['action'], 'keep')
            self.assertEqual(by_path[str(data / 'large.bin')]['reason'], 'model-or-dataset')
            self.assertEqual(by_path[str(work)]['action'], 'blocked')
            self.assertEqual(by_path[str(root / 'other-user')]['reason'], 'different-owner')

    def test_local_sources_block_containing_paths_and_active_jobs_block_all(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'build/custom.cu'
            audit = dict(root=str(root), tracked_changes=[], unpushed=[], stash=[],
                         ignored_source=[str(source)])
            data = dict(repository=str(root), worktrees=[audit])
            rows = [row(root / 'build', 'B')]
            result = cleanup.plan(data, rows, owner='developer')
            self.assertEqual(result[0]['reason'], 'untracked-source')
            result = cleanup.plan(data, rows, active_jobs=['numeric'], owner='developer')
            self.assertEqual(result[0]['reason'], 'project-jobs-active')

    def test_symlink_and_old_incomplete_audits_require_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            (root / 'link').symlink_to(root / 'data', target_is_directory=True)
            result = cleanup.plan(dict(repository=str(root), worktrees=[]),
                                  [row(root / 'link')], owner='developer')
            self.assertEqual(result[0]['reason'], 'symlink-review')
            audit = dict(root=str(root), unpushed=[], stash=[])
            result = cleanup.plan(dict(repository=str(root), worktrees=[audit]),
                                  [row(root / 'build', 'B')], owner='developer')
            self.assertEqual(result[0]['reason'], 'unpublished-code-or-incomplete-audit')

    def test_inventory_output_cannot_overwrite_tracked_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, 'ignored runs/'):
                inventory.collect(root, root / 'docs/results')


if __name__ == '__main__':
    unittest.main()
