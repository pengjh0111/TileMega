import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'docs_dependency_audit', ROOT / 'scripts/audit_docs_dependencies.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
HISTORY = '/'.join(('docs', 'experiments'))


class DocsDependencyAuditTest(unittest.TestCase):
    def test_joined_and_dynamic_references(self):
        self.assertEqual(audit.references("root / 'docs' / 'fixtures' / 'input.json'"),
                         [('docs/fixtures/input.json', False)])
        self.assertEqual(audit.references("Path(root, 'docs', 'fixtures') / name"),
                         [('docs/fixtures', True)])
        self.assertEqual(audit.references("root / 'docs' / 'fixtures' / name"),
                         [('docs/fixtures', True)])
        self.assertEqual(audit.references('root + "/docs/fixtures/" + model + ".json"'),
                         [('/docs/fixtures/', True)])
        self.assertEqual(audit.references('${ROOT}/docs/${MODEL}/input.json'),
                         [('/docs/${MODEL}/input.json', True)])

    def test_inventory_preserves_inputs_and_distinguishes_local_fixtures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            paths = {
                'test/case.cpp': '\n'.join((
                    f'auto a = root + "/{HISTORY}/topic/tracked.json";',
                    f'auto b = root + "/{HISTORY}/topic/local.json";',
                    f'auto c = root + "/{HISTORY}/topic/missing.json";',
                    f'auto d = root + "/{HISTORY}/topic/" + model + ".json";',
                    f'// Review {HISTORY}/topic/note.md',
                    'auto help = "docs/BUILD_MLIR.md";',
                )),
                'configs/e2e/example.json':
                    '{"pin": "' + str(root / HISTORY / 'topic/tracked.json') + '"}',
                'configs/targets/example.json':
                    '{"source": "' + HISTORY + '/topic/old_target.json"}',
                'scripts/joined.py': "source = root / 'docs' / 'fixtures' / 'input.json'\n",
                f'{HISTORY}/topic/tracked.json': '{}\n',
                'docs/fixtures/input.json': '{}\n',
                'docs/BUILD_MLIR.md': 'kept\n',
            }
            for name, content in paths.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Audit', '-c',
                            'user.email=audit@example.invalid', 'commit', '-qm', 'fixture'], check=True)
            local = root / HISTORY / 'topic/local.json'
            local.write_text('{"local": true}\n')
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file() and '.git' not in p.parts}
            with contextlib.redirect_stdout(io.StringIO()):
                summary, rows = audit.collect(root, root / 'runs/r15/dependencies')
            by_line = {r['line']: r for r in rows if r['source'] == 'test/case.cpp'}
            self.assertEqual([by_line[n]['status'] for n in range(1, 5)],
                             ['tracked_file', 'untracked_file', 'missing', 'dynamic_review'])
            self.assertEqual(by_line[4]['tracked_children'], 1)
            self.assertEqual(by_line[5]['kind'], 'comment')
            self.assertEqual(by_line[6]['kind'], 'retained_document')
            provenance = next(r for r in rows if r['source'] == 'configs/targets/example.json')
            self.assertEqual(provenance['kind'], 'provenance')
            absolute = next(r for r in rows if r['source'] == 'configs/e2e/example.json')
            self.assertEqual(absolute['status'], 'tracked_file')
            self.assertEqual(absolute['relative'], f'{HISTORY}/topic/tracked.json')
            self.assertEqual(summary['references'], 9)
            self.assertEqual(summary['mode'], 'read-only')
            for path, original in before.items():
                self.assertEqual(path.read_bytes(), original)
            self.assertTrue((root / 'runs/r15/dependencies/dependencies.tsv').is_file())

    def test_external_paths_are_not_opened_and_output_stays_in_runs(self):
        row = audit.inspect(Path('/repository'), [], 'configs/model.json', 1,
            '{}', '/someone-else/docs/target.json', False)
        self.assertEqual(row['status'], 'external_path')
        self.assertFalse(row['exists'])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, 'ignored runs/'):
                audit.collect(root, root / 'docs/results/raw-audit')


if __name__ == '__main__':
    unittest.main()
