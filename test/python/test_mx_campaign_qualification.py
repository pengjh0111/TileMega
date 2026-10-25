"""Exact domain qualification uses CPU fixtures and real aggregation validation."""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


source = Path(__file__).resolve().parents[2] / 'experiments/mx5/qualify_campaign.py'
spec = importlib.util.spec_from_file_location('mx5_qualify_campaign', source)
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)

fixture_spec = importlib.util.spec_from_file_location('mx5_qualification_fixtures',
                                                     Path(__file__).with_name('test_mx_stage_table.py'))
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


class CampaignQualification(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inventory = self.root / 'inventory.jsonl'
        self.checked_path = self.root / 'checked.jsonl'
        self.manifest = self.root / 'campaign.json'
        self.checked, self.inventory_rows, self.jobs = [], [], []
        self.measurements = {}
        for index, kind in enumerate(qualifier.KINDS):
            row = fixtures.measurement(kind)
            row['key']['threads'] = 512
            key = qualifier.table.canonical(row, kind)
            ref = fixtures.reference(row)
            use = [dict(model='model_a', stage=kind), dict(model='model_b', stage=kind)]
            resources = {field: row[field] for field in qualifier.table.RESOURCE_FIELDS}
            checked = dict(kind=kind, key=key, library='/fixture/' + kind + '.so', uses=use,
                           query_only=True, query_status=0, fits=True, selected_resource_step=0,
                           provenance=qualifier.table.identity(row), resources=resources,
                           r_next_reference=ref)
            self.checked.append(checked)
            base = dict(key)
            del base['prefetch_kind' if kind == 'attention' else 'prefetch']
            self.inventory_rows.append(dict(kind=kind, key=base, uses=list(reversed(use))))
            directory = self.root / kind
            directory.mkdir()
            key_file = directory / 'keys.json'
            key_file.write_text(json.dumps([dict(key, library=checked['library'])]))
            write_rows(directory / 'checked.jsonl', [checked])
            (directory / 'references.json').write_text(json.dumps(dict(references=[ref])))
            job = dict(kind=kind, keys=str(key_file), directory=str(directory), rows=1,
                       keys_sha256=qualifier.sha(key_file.read_bytes()))
            self.jobs.append(job)
            self.measurements[kind] = dict(row, key=json.loads(key_file.read_text())[0])
            for round_number in (1, 2, 3):
                output = directory / f'round{round_number}.jsonl'
                write_rows(output, [dict(self.measurements[kind], round=round_number)])
                output.with_suffix('.log').write_text('fixture success\n')
                (directory / f'process{round_number}.json').write_text(json.dumps(dict(
                    round=round_number, returncode=0,
                    command=['python', 'bench.py', '--keys', str(key_file), '--out', str(output),
                             '--round', str(round_number)])))
            guard = directory / 'guard'
            guard.mkdir()
            (guard / 'guard_result.json').write_text(json.dumps(dict(code=0)))
            write_rows(guard / 'guard.jsonl', [dict(phase='after', exit_code=0)])
            (directory / 'scheduler_job.json').write_text(json.dumps(dict(
                card=7 if index % 2 == 0 else 6, keys_sha256=job['keys_sha256'],
                command=['guard.sh', '7' if index % 2 == 0 else '6', '900', str(guard)])))
        write_rows(self.inventory, self.inventory_rows)
        self.refresh_manifest()
        self.args = argparse.Namespace(manifest=[self.manifest], inventory=self.inventory,
                                       threads=512, job_log=self.root / 'gpu_jobs.log',
                                       check_only=False)

    def refresh_manifest(self):
        write_rows(self.checked_path, self.checked)
        self.manifest.write_text(json.dumps(dict(processes_per_key=3, rows=len(self.jobs),
                                                checked=str(self.checked_path),
                                                checked_sha256=qualifier.sha(self.checked_path.read_bytes()),
                                                jobs=self.jobs)))

    def test_exact_domain_qualifies_scoped_table_with_actual_cards_and_sources(self):
        document, code = qualifier.qualify(self.args)
        self.assertEqual(code, 0)
        self.assertTrue(document['qualified'])
        self.assertTrue(document['measured_domain_complete'])
        self.assertFalse(document['full_table'])
        self.assertFalse(document['global_dt576_policy_approved'])
        self.assertEqual(document['coverage_threads'], [512])
        self.assertEqual(document['coverage']['expected_keys'], 3)
        self.assertEqual({row['kind']: row['card'] for row in document['rows']},
                         dict(gemm='7', head='6', attention='7'))
        self.assertEqual([row['r_next'] for row in document['rows']], [72, 72, 72])
        self.assertTrue(all(row['resource_selection'] == {'selected_resource_step': 0}
                            for row in document['rows']))
        for snapshot in document['provenance']['tool_sources']:
            self.assertEqual(hashlib.sha256(snapshot['source'].encode()).hexdigest(), snapshot['sha256'])
        self.assertEqual(document['provenance']['manifests'][0]['sha256'],
                         qualifier.sha(self.manifest.read_bytes()))

    def test_prefetch_reduction_attempts_survive_table_publication(self):
        checked = self.checked[0]
        attempts = [dict(resource_step=0, prefetch=4, fits=False),
                    dict(resource_step=1, prefetch=2, fits=True)]
        checked.update(selected_resource_step=1, resource_attempts=attempts)
        write_rows(Path(self.jobs[0]['directory']) / 'checked.jsonl', [checked])
        self.refresh_manifest()
        document, _ = qualifier.qualify(self.args)
        gemm = next(row for row in document['rows'] if row['kind'] == 'gemm')
        self.assertEqual(gemm['resource_selection'],
                         dict(selected_resource_step=1, resource_attempts=attempts))

    def test_check_only_accepts_coverage_without_timing_and_never_qualifies(self):
        self.args.check_only = True
        (Path(self.jobs[0]['directory']) / 'round1.jsonl').unlink()
        with patch.object(qualifier.scheduler, 'batch_state', side_effect=AssertionError('timings read')):
            document, code = qualifier.qualify(self.args)
        self.assertEqual(code, 0)
        self.assertTrue(document['coverage']['exact'])
        self.assertFalse(document['qualified'])
        self.assertFalse(document['full_table'])
        self.assertFalse(document['measured_domain_complete'])

    def test_multiple_manifests_combine_without_duplicate_domain_keys(self):
        first = json.loads(self.manifest.read_text())
        second = dict(first, jobs=first['jobs'][1:], rows=2)
        first.update(jobs=first['jobs'][:1], rows=1)
        self.manifest.write_text(json.dumps(first))
        other = self.root / 'campaign2.json'
        other.write_text(json.dumps(second))
        self.args.manifest.append(other)
        document, code = qualifier.qualify(self.args)
        self.assertEqual(code, 0)
        self.assertTrue(document['measured_domain_complete'])
        self.assertEqual(len(document['provenance']['manifests']), 2)

    def test_actual_prefetch_cannot_be_projected_out_of_measurement_validation(self):
        directory = Path(self.jobs[0]['directory'])
        row = copy.deepcopy(self.measurements['gemm'])
        row['key']['prefetch'] = 2
        row['prefetch'] = 2
        write_rows(directory / 'round1.jsonl', [row])
        with self.assertRaisesRegex(ValueError, 'partial or unsuccessful'):
            qualifier.qualify(self.args)

    def test_missing_and_extra_domain_keys(self):
        inventory = copy.deepcopy(self.inventory_rows)
        inventory[0]['key']['b'] = 2
        write_rows(self.inventory, inventory)
        with self.assertRaisesRegex(ValueError, 'missing=1 extra=1'):
            qualifier.qualify(self.args)
        self.args.check_only = True
        document, code = qualifier.qualify(self.args)
        self.assertEqual(code, 1)
        self.assertEqual((document['coverage']['missing_keys'], document['coverage']['extra_keys']), (1, 1))
        self.assertFalse(document['measured_domain_complete'])

    def test_duplicate_inventory_and_campaign_base_keys_rejected(self):
        write_rows(self.inventory, [*self.inventory_rows, self.inventory_rows[0]])
        with self.assertRaisesRegex(ValueError, 'duplicate inventory'):
            qualifier.qualify(self.args)
        write_rows(self.inventory, self.inventory_rows)
        self.jobs.append(self.jobs[0])
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate campaign base key'):
            qualifier.qualify(self.args)

    def test_checked_source_hash_and_local_rows_are_immutable(self):
        raw = self.checked_path.read_bytes()
        self.checked_path.write_bytes(raw + b'\n')
        with self.assertRaisesRegex(ValueError, 'checked source changed'):
            qualifier.qualify(self.args)
        self.checked_path.write_bytes(raw)
        directory = Path(self.jobs[0]['directory'])
        changed = copy.deepcopy(self.checked[0])
        changed['disposition'] = 'tampered'
        write_rows(directory / 'checked.jsonl', [changed])
        with self.assertRaisesRegex(ValueError, 'differs from immutable'):
            qualifier.qualify(self.args)

    def test_uses_are_exact_sets(self):
        self.inventory_rows[0]['uses'].append(dict(model='extra', stage='gemm'))
        write_rows(self.inventory, self.inventory_rows)
        with self.assertRaisesRegex(ValueError, 'uses differ'):
            qualifier.qualify(self.args)

    def test_failed_guard_or_incomplete_process_rejected(self):
        directory = Path(self.jobs[0]['directory'])
        guard = directory / 'guard/guard_result.json'
        original = guard.read_text()
        guard.write_text(json.dumps(dict(code=75)))
        with self.assertRaisesRegex(ValueError, 'partial or unsuccessful'):
            qualifier.qualify(self.args)
        guard.write_text(original)
        (directory / 'process3.json').unlink()
        with self.assertRaisesRegex(ValueError, 'partial or unsuccessful'):
            qualifier.qualify(self.args)

    def test_measurement_must_match_checked_source_and_real_prefetch(self):
        directory = Path(self.jobs[0]['directory'])
        for number in (1, 2, 3):
            changed = dict(self.measurements['gemm'], round=number, source_sha256='e' * 64)
            write_rows(directory / f'round{number}.jsonl', [changed])
        with self.assertRaisesRegex(ValueError, 'R_next reference identity conflict'):
            qualifier.qualify(self.args)

    def test_manual_pilot_card_uses_resolved_guard_path_and_success_pair(self):
        job = self.jobs[0]
        directory = Path(job['directory'])
        (directory / 'scheduler_job.json').unlink()
        guard = directory / 'guard'
        # A command can mention another out=/rc=; only the outer fields count.
        self.args.job_log.write_text(
            f'2026-10-25T00:00:00 gpu=6 timeout=900 out={guard} cmd=echo rc=9 out=/unrelated\n'
            f'2026-10-25T00:00:01 gpu=6 rc=0 out={guard}\n')
        document, _ = qualifier.qualify(self.args)
        self.assertEqual(document['rows'][0]['card'], '6')
        evidence = document['provenance']['batches'][0]['card_evidence']
        self.assertEqual(len(evidence['matched_lines']), 2)
        with self.args.job_log.open('a') as stream:
            stream.write(f'2026-10-25T00:00:02 gpu=7 rc=0 out={guard}\n')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            qualifier.qualify(self.args)

    def test_missing_card_evidence_never_defaults_to_zero(self):
        (Path(self.jobs[0]['directory']) / 'scheduler_job.json').unlink()
        self.args.job_log.write_text('')
        with self.assertRaisesRegex(ValueError, 'card provenance'):
            qualifier.qualify(self.args)

    def test_output_is_created_exclusively(self):
        output = self.root / 'qualification.json'
        argv = ['qualify_campaign.py', '--manifest', str(self.manifest),
                '--inventory', str(self.inventory), '--threads', '512', '--check-only', '--out', str(output)]
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(qualifier.main(), 0)
        original = output.read_bytes()
        with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                qualifier.main()
        self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
