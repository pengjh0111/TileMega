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
        self.write_csv(primary, [dict(model=m, batch=b, arm=a, status='N/A' if m == 'nafnet' and a == 4 else 'pass')
            for m,b in controller.CELLS for a in range(5)])
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
