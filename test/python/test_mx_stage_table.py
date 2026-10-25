"""Measured cost rows must retain exact coordinates and three-process evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

source = Path(__file__).resolve().parents[2] / 'experiments/mx5/stage_table.py'
spec = importlib.util.spec_from_file_location('mx5_stage_table', source)
table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table)


def measurement(kind='gemm'):
    keys = {
        'gemm': dict(n=32768, k=1024, b=1, tn=16, wn=1, depth=1, op=0, norm=1, part=0, pw=1),
        'head': dict(n=32768, k=1024, b=1, tn=16, wn=4, depth=1, pw=1),
        'attention': dict(dim=64, q=2, hkv=4, b=1, length=65, b_min=1, qk_norm=0, prefetch_kind=2),
    }
    row = dict(key=keys[kind], round=1, executor='mx_static_stage_bench', implementation=f'mx5_{kind}_v2',
               median_ns=100, p90_ns=150, iterations=200, copies=8, working_set_bytes=256 * 1024**2,
               mtreg=80, local_bytes=0, static_smem_bytes=0, resident_ctas=1, threads=512,
               source_sha256='a' * 64, binary_sha256='b' * 64, base_head='c' * 40)
    row['prefetch_kind' if kind == 'attention' else 'prefetch'] = 2 if kind == 'attention' else 1
    return row


def reference(row):
    kind = table.kind_of(row)
    key = table.canonical(row, kind)
    key['prefetch_kind' if kind == 'attention' else 'prefetch'] = 0
    return dict(kind=kind, key=key, resources=dict(mtreg=72, local_bytes=0, static_smem_bytes=0, resident_ctas=1),
                provenance=dict(table.identity(row), binary_sha256='d' * 64))


class StageTable(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def runs(self, rows=None):
        rows = rows or [measurement()]
        paths = []
        for round_number, median in [(1, 90), (2, 100), (3, 120)]:
            values = [dict(copy.deepcopy(row), round=round_number, median_ns=median,
                           p90_ns=median + 50) for row in rows]
            path = self.directory / f'round{round_number}.jsonl'
            path.write_text(''.join(json.dumps(value) + '\n' for value in values))
            paths.append(path)
        return paths

    def edit(self, path, update):
        values = [json.loads(line) for line in path.read_text().splitlines()]
        update(values)
        path.write_text(''.join(json.dumps(value) + '\n' for value in values))

    def test_three_body_types_and_process_statistics(self):
        rows = [measurement(kind) for kind in ('gemm', 'head', 'attention')]
        for row in rows:
            row['r_next_reference'] = reference(row)
        document = table.aggregate(self.runs(rows), card='C550 GPU 7')
        self.assertTrue(document['qualified'])
        self.assertFalse(document['full_table'])
        self.assertEqual({row['kind'] for row in document['rows']}, {'gemm', 'head', 'attention'})
        for row in document['rows']:
            self.assertEqual((row['median_ns'], row['p90_ns'], row['spread']), (100, 150, .3))
            self.assertEqual(row['r_next'], 72)
            self.assertEqual([sample['round'] for sample in row['rounds']], [1, 2, 3])
            self.assertEqual(row['key']['threads'], 512)

    def test_diagnostic_is_unqualified_and_omits_rnext(self):
        document = table.aggregate(self.runs(), diagnostic=True)
        self.assertFalse(document['qualified'])
        self.assertFalse(document['full_table'])
        self.assertNotIn('r_next', document['rows'][0])
        with self.assertRaisesRegex(table.TableError, 'missing R_next'):
            table.aggregate(self.runs(), card='C550 GPU 7')

    def test_missing_and_duplicate_rounds(self):
        paths = self.runs()
        with self.assertRaisesRegex(table.TableError, 'missing round'):
            table.aggregate(paths[:2], diagnostic=True)
        self.edit(paths[1], lambda rows: rows.append(copy.deepcopy(rows[0])))
        with self.assertRaisesRegex(table.TableError, 'duplicate round'):
            table.aggregate(paths, diagnostic=True)
        with self.assertRaisesRegex(table.TableError, 'duplicate input'):
            table.aggregate([paths[0], paths[0]], diagnostic=True)

    def test_one_round_per_process_file(self):
        paths = self.runs()
        self.edit(paths[0], lambda rows: rows.append(dict(rows[0], round=2)))
        with self.assertRaisesRegex(table.TableError, 'one round'):
            table.aggregate(paths, diagnostic=True)

    def test_coordinate_conflicts_and_unknown_static_shape(self):
        for changes in [dict(threads=576), dict(pw=2), dict(depth=True), dict(static_shape=True)]:
            with self.subTest(changes=changes):
                row = measurement()
                row['key'].update(changes)
                with self.assertRaises(table.TableError):
                    table.aggregate(self.runs([row]), diagnostic=True)

    def test_round_identity_and_resource_conflicts(self):
        for field, value in [('binary_sha256', 'e' * 64), ('source_sha256', 'f' * 64), ('mtreg', 81)]:
            with self.subTest(field=field):
                paths = self.runs()
                self.edit(paths[1], lambda rows: rows[0].update({field: value}))
                with self.assertRaisesRegex(table.TableError, 'conflict'):
                    table.aggregate(paths, diagnostic=True)

    def test_resource_caps_reject_spills_even_in_diagnostics(self):
        for changes in [dict(local_bytes=4), dict(threads=512, mtreg=256), dict(threads=576, mtreg=228)]:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(table.TableError, 'resource rejection'):
                    table.aggregate(self.runs([dict(measurement(), **changes)]), diagnostic=True)

    def test_geometry_wait_and_stage_resource_invariants(self):
        for kind, updates in [('gemm', dict(n=16 * 104, wn=2, part=1)),
                              ('head', dict(n=16 * 104)),
                              ('gemm', dict(n=16, k=32)),
                              ('gemm', dict(pw=64))]:
            with self.subTest(kind=kind, updates=updates):
                row = measurement(kind)
                row['key'].update(updates)
                if 'pw' in updates:
                    row['prefetch'] = updates['pw']
                with self.assertRaises(table.TableError):
                    table.aggregate(self.runs([row]), diagnostic=True)
        for changes in [dict(resident_ctas=2), dict(static_smem_bytes=16)]:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(table.TableError, 'one resident CTA'):
                    table.aggregate(self.runs([dict(measurement(), **changes)]), diagnostic=True)

    def test_resource_reference_mapping_and_exact_body_identity(self):
        row = measurement()
        path = self.directory / 'resources.json'
        ref = reference(row)
        ref['provenance']['base_head'] = 'e' * 40
        path.write_text(json.dumps(dict(references=[ref])))
        result = table.aggregate(self.runs([row]), [path], card='C550 GPU 7')
        self.assertEqual(result['rows'][0]['r_next'], 72)
        for name, value in [('op', 1), ('norm', 0), ('part', 1), ('wn', 2), ('threads', 576), ('prefetch', 1)]:
            with self.subTest(name=name):
                wrong = copy.deepcopy(ref)
                wrong['key'][name] = value
                row['r_next_reference'] = wrong
                with self.assertRaises(table.TableError):
                    table.aggregate(self.runs([row]), card='C550 GPU 7')
        for change in [dict(source_sha256='f' * 64), dict(implementation='mx5_head_v2')]:
            wrong = copy.deepcopy(ref)
            wrong['provenance'].update(change)
            row['r_next_reference'] = wrong
            with self.assertRaises(table.TableError):
                table.aggregate(self.runs([row]), card='C550 GPU 7')
        for value in (0, 256):
            wrong = copy.deepcopy(ref)
            wrong['resources']['mtreg'] = value
            row['r_next_reference'] = wrong
            with self.assertRaises(table.TableError):
                table.aggregate(self.runs([row]), card='C550 GPU 7')

    def test_legacy_depth_uses_matching_build_metadata(self):
        row = measurement()
        del row['key']['depth']
        library = self.directory / 'stage.so'
        row['key']['library'] = str(library)
        meta = dict(table.identity(row), body='gemm', bench=True, returncode=0, tn=16, depth=1, threads=512, pw=1)
        library.with_suffix('.json').write_text(json.dumps(meta))
        result = table.aggregate(self.runs([row]), diagnostic=True)
        self.assertEqual(result['rows'][0]['key']['depth'], 1)
        meta['binary_sha256'] = 'e' * 64
        library.with_suffix('.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(table.TableError, 'metadata identity conflict'):
            table.aggregate(self.runs([row]), diagnostic=True)


if __name__ == '__main__':
    unittest.main()
