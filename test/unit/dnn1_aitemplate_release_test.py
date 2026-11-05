"""CPU-only release lineage fixtures; no scheduler or CUDA initialization."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import dnn1_aitemplate_collect as controller


class ReleaseTest(unittest.TestCase):
    def test_dispatcher_startup_state_waits_without_releasing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            self.assertEqual(controller.tvm_state(path)['status'], 'state_initializing')
            path.write_text('{')
            self.assertEqual(controller.tvm_state(path)['status'], 'state_initializing')
            path.write_text('{"status":"collecting"}')
            self.assertEqual(controller.tvm_state(path)['status'], 'collecting')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tvm = self.root / 'runs/dnn1/resume_tvm/resumed'
        self.tvm.mkdir(parents=True)
        self.state = {'status': 'primary_complete', 'workers': {}, 'active': {}}
        controller.base.write(self.tvm / 'state.json', self.state)
        self.identity = {'recipe': {'graph_mode': True}, 'source': 'fixture-source'}
        proof = self.root / 'cpu.json'
        controller.base.write(proof, dict(identity=self.identity,
            rows=[dict(model=m, batch=b, status='ir_created') for m,b in controller.CELLS]))
        primary = self.root / 'primary.csv'
        self.write_csv(primary, [dict(model=m, batch=b, arm=a,
            system='tensorrt' if a < 2 else 'torchcompile',
            status='N/A' if m == 'nafnet' and a == 4 else 'pass')
            for m,b in controller.CELLS for a in range(5)] +
            [dict(model=m, batch=b, arm='tvm-bf16', system='tvm', status='pass')
             for m,b in controller.CELLS])
        selected = []
        for m,b in sorted(controller.CELLS):
            folder = self.tvm / f'{m}_{b}'; folder.mkdir()
            files = [folder / f'{i}.json' for i in range(6)]
            for path in files:
                controller.base.write(path, {'versions': {'source_version': 'fixture'}})
            selected.append(dict(model=m, batch=b, status='pass', correctness_passes='5/5',
                source=json.dumps(list(map(str, files[1:]))), cell_session=str(files[0])))
        self.write_csv(self.tvm / 'frozen20.csv', selected[:20])
        self.write_csv(self.tvm / 'accepted_delta.csv', selected[20:])
        self.config = dict(tvm_epoch=str(self.tvm), cpu_prepare=str(proof),
            frozen20_sha256=controller.base.sha(self.tvm / 'frozen20.csv'),
            primary_csv=str(primary), primary_sha256=controller.base.sha(primary))
        self.out = self.root / 'ait'; self.out.mkdir()

    def write_csv(self, path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def release(self):
        with patch.object(controller, 'ROOT', self.root), \
             patch.object(controller.runner, 'ROOT', self.root), \
             patch.object(controller.runner, 'source_identity', return_value=self.identity):
            return controller.release(self.config, self.out)

    def test_full25_and_drained_repairs_release_versioned_evidence(self):
        value = self.release()
        self.assertIn('TVM25', value['policy'])
        ledger = controller.base.read(self.root / 'runs/dnn1/resume_tvm/aitemplate_release.json')
        self.assertEqual(len(ledger['cells']), 25)
        self.assertTrue(ledger['graph_mode'])

    def test_explicit_release_reuse_binds_new_source_without_rewriting_TVM_ledger(self):
        self.release()
        ledger = self.root / 'runs/dnn1/resume_tvm/aitemplate_release.json'
        digest = controller.base.sha(ledger)
        self.config['reuse_release_sha256'] = digest
        self.out = self.root / 'retry'
        value = self.release()
        self.assertEqual(controller.base.sha(ledger), digest)
        self.assertTrue(Path(value['current_source_binding']).is_file())

    def test_implicit_release_reuse_cannot_overwrite_prior_ledger(self):
        self.release()
        self.out = self.root / 'retry'
        with self.assertRaises(FileExistsError):
            self.release()

    def test_finalizer_also_requires_original_dispatcher_drained(self):
        upstream = self.root / 'upstream.json'
        controller.base.write(upstream, {'status':'needs_diagnosis', 'workers':{}})
        self.config['additional_dispatchers'] = [str(upstream)]
        self.release()
        ledger = controller.base.read(self.root / 'runs/dnn1/resume_tvm/aitemplate_release.json')
        self.assertEqual(len(ledger['dispatchers']),2)
        self.assertIn(str(upstream),ledger['dispatchers'])

    def split_epochs(self):
        other=self.root/'second';other.mkdir()
        selected=controller.rows(self.tvm/'accepted_delta.csv')
        self.write_csv(self.tvm/'accepted_delta.csv',selected[:4])
        self.write_csv(other/'accepted_delta.csv',selected[4:])
        (other/'frozen20.csv').write_bytes((self.tvm/'frozen20.csv').read_bytes())
        controller.base.write(self.tvm/'state.json',{'status':'needs_diagnosis','workers':{},'accepted':4})
        controller.base.write(other/'state.json',{'status':'primary_complete','workers':{},'accepted':1})
        self.config['tvm_epochs']=[str(self.tvm),str(other)]
        return other

    def test_split_epoch_keeps_all25_distinct_cells_and_both_drain_gates(self):
        self.split_epochs();self.release()
        ledger=controller.base.read(self.root/'runs/dnn1/resume_tvm/aitemplate_release.json')
        self.assertEqual(len(ledger['cells']),25)
        self.assertEqual(len(ledger['dispatchers']),2)

    def test_duplicate_across_epochs_is_rejected_without_picking_a_winner(self):
        other=self.split_epochs()
        selected=controller.rows(self.tvm/'accepted_delta.csv')
        self.write_csv(other/'accepted_delta.csv',[selected[0]])
        with self.assertRaisesRegex(ValueError,'distinct'):self.release()

    def test_split_epoch_never_releases_while_one_payload_remains(self):
        other=self.split_epochs()
        controller.base.write(other/'state.json',{'status':'collecting','workers':{'5':{'pid':123}}})
        with self.assertRaisesRegex(RuntimeError,'active'):self.release()

    def test_active_repair_never_releases(self):
        self.state['workers'] = {'5': {'pid': 123}}
        controller.base.write(self.tvm / 'state.json', self.state)
        with self.assertRaisesRegex(RuntimeError, 'active'): self.release()

    def test_cpu_proof_of_other_source_never_releases(self):
        self.identity = {'source': 'different-source'}
        with self.assertRaisesRegex(ValueError, 'CPU25'): self.release()

    def test_changed_frozen20_never_releases(self):
        with (self.tvm / 'frozen20.csv').open('a') as stream: stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'frozen20'): self.release()

    def test_duplicate_cell_never_releases(self):
        selected = controller.rows(self.tvm / 'accepted_delta.csv')
        selected[-1] = selected[0]
        self.write_csv(self.tvm / 'accepted_delta.csv', selected)
        with self.assertRaisesRegex(ValueError, 'distinct'): self.release()

    def test_missing_fifth_process_evidence_never_releases(self):
        selected = controller.rows(self.tvm / 'accepted_delta.csv')
        selected[0]['source'] = json.dumps(json.loads(selected[0]['source'])[:4])
        self.write_csv(self.tvm / 'accepted_delta.csv', selected)
        with self.assertRaisesRegex(ValueError, 'five receipts'): self.release()


if __name__ == '__main__':
    unittest.main()
