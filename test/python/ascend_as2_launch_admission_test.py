"""CPU admission fixtures never initialize a device or enter formal targets."""
import functools
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
from tilemega.bench import as1_run
from tilemega.bench.as2_run import model_topology_preflight


class PhysicalAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cpu_admission_',
            dir='/home/d50064638/junhuip/TileMega/runs/as2/probes')
        self.root = Path(self.temp.name)
        self.repo = self.root/'TileMega'
        self.runs = self.repo/'runs/as2'
        self.binding = self.runs/'binding'
        self.binding.mkdir(parents=True)
        self.target = self.repo/'configs/targets/cpu_fixture.json'
        self.target.parent.mkdir(parents=True)
        self.proof = self.runs/'cpu_fixture_proof.json'
        self.proof.write_text(json.dumps(dict(status='pass', physical_device_id=0,
            processes_passed_per_group=5, groups=[14], cpu_fixture=True)))
        self.target.write_text(json.dumps(dict(arch_tag='ascend_3510', product_sku='950PR',
            max_verified_mix_groups=14, verified_physical_device_ids=[0],
            topology_experiment=dict(source_summary=str(self.proof),
                source_summary_sha256=self.digest(self.proof)))))
        phases = {}
        for phase in ('prefill', 'decode'):
            plan = self.runs/('source_'+phase)/'plan'
            plan.mkdir(parents=True)
            (plan.parent/'import.json').write_text(json.dumps(dict(target_config=str(self.target),
                target_config_sha256=self.digest(self.target))))
            bound = self.binding/phase
            bound.mkdir()
            bound.joinpath('metadata.bin').write_bytes(struct.pack('<12I3Q',
                4,14,42,1,1,0,1,2,16,64 if phase=='prefill' else 1,1088,0,9<<30,131072,2000000000))
            phases[phase] = dict(source_plan=str(plan))
        (self.binding/'request.json').write_text(json.dumps(dict(groups=14, phases=phases)))
        self.command = [str(self.root/'tilemega-as2/build/test/as2_llm'), 'request', str(self.binding),
                        str(self.runs/'output'), '67', '0', '1', '0']

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def check(self, physical):
        return model_topology_preflight(self.command, physical, repo=self.repo)

    def test_g14_requires_the_actual_binary_topology_and_physical_proof(self):
        self.assertEqual(self.check(0)['status'], 'checked')
        self.assertIn('physical-device1', self.check(1)['reason'])
        receipt = self.binding/'request.json'
        value = json.loads(receipt.read_text());value['groups']=8
        receipt.write_text(json.dumps(value))
        self.assertEqual(self.check(1)['status'], 'blocked')
        self.proof.write_text(self.proof.read_text()+'\n')
        self.assertIn('Changed G14', self.check(0)['reason'])

    def test_existing_g8_and_non_model_probes_keep_their_scope(self):
        receipt=self.binding/'request.json'
        identity=json.loads(receipt.read_text());identity['groups']=8
        receipt.write_text(json.dumps(identity))
        for phase in ('prefill','decode'):
            path = self.binding/phase/'metadata.bin'
            values = list(struct.unpack('<12I3Q',path.read_bytes()));values[1:3]=[8,24]
            path.write_bytes(struct.pack('<12I3Q',*values))
        self.assertEqual(self.check(0)['status'], 'checked')
        self.assertEqual(self.check(1)['status'], 'checked')
        path = self.binding/'decode/metadata.bin'
        values = list(struct.unpack('<12I3Q',path.read_bytes()));values[1:3]=[14,42]
        path.write_bytes(struct.pack('<12I3Q',*values))
        self.assertIn('different model topologies',self.check(0)['reason'])
        self.assertEqual(model_topology_preflight(['as2_message_passing','14','output'],1)['status'],
                         'not_applicable')

    def test_rejected_model_never_spawns_a_child(self):
        exe=Path(self.command[0]);exe.parent.mkdir(parents=True);exe.write_text('CPU fixture; never execute')
        out=self.runs/'jobs/rejected'
        argv=['as2_run','--physical-device','1','--out',str(out),'--',*self.command]
        with mock.patch.multiple(as1_run,ROOT=self.root,WORK=self.root/'tilemega-as2',RUNS=self.runs), \
             mock.patch.object(as1_run,'snapshot',return_value=dict(exit_code=0,output='')), \
             mock.patch.object(as1_run,'devices_from_snapshot',return_value={1:dict(health='OK',free_mib=1000)}), \
             mock.patch.object(as1_run.subprocess,'Popen') as child, mock.patch('sys.argv',argv):
            result=as1_run.main(project='as2',command_preflight=functools.partial(
                model_topology_preflight,repo=self.repo))
        self.assertEqual(result,78);child.assert_not_called()
        receipt=json.loads((out/'job.json').read_text())
        self.assertEqual(receipt['status'],'blocked-capability');self.assertNotIn('pid',receipt)


if __name__=='__main__':unittest.main()
