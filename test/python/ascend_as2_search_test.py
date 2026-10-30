"""CPU-only bounded search guards; synthetic costs never reach a target."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tilemega.bench.as2_models import RUNS
from tilemega.bench.as2_search import select,validate


class AscendOuterSearch(unittest.TestCase):
    def test_outer_search_rejects_incomparable_calibration_before_solving(self):
        with tempfile.TemporaryDirectory(dir=RUNS/'probes') as raw:
            root=Path(raw);export=root/'export';export.mkdir()
            identity=dict(model_id='cpu-fixture',config_sha256='fixture',phase='decode')
            (export/'as2_export.json').write_text(json.dumps(identity))
            def candidate(name,device=1,calibration='same'):
                p=root/name;p.mkdir()
                job=dict(status='passed',physical_device_id=device,executable_sha256='fixture-binary')
                (p/'job.json').write_text(json.dumps(job))
                cost=dict(status='measured_empirical_context',groups=8,physical_device_id=device,cpu_fixture=calibration)
                profile=dict(status='measured_body_only',groups=8,physical_device_id=device,job=str(p),binary_sha256='fixture-binary')
                for n,v in [('cost',cost),('profile',profile)]:
                    (p/(n+'.json')).write_text(json.dumps(v))
                price=dict(identity,schema='tilemega.as2.native_task_prices.v1',status='sourced_empirical_context',
                    batch=16,groups=8,runtime_abi=4,past_min=0,past_max=1086,
                    source_profile=str(p/'profile.json'),source_costs=str(p/'cost.json'),
                    source_profile_sha256=hashlib.sha256((p/'profile.json').read_bytes()).hexdigest(),
                    source_costs_sha256=hashlib.sha256((p/'cost.json').read_bytes()).hexdigest())
                (p/'price.json').write_text(json.dumps(price))
                return dict(id=name,groups=8,gemm_tile_n=128,kv_block=1088,attention='cpu-fixture-'+name,prices=str(p/'price.json'))
            a=candidate('a');b=candidate('b')
            domain=dict(schema='tilemega.as2.outer_search_domain.v1',export=str(export),batch=16,candidates=[a,b])
            self.assertEqual(validate(domain)[1]['model_id'],'cpu-fixture')
            with self.assertRaisesRegex(ValueError,'Duplicate implementation'):
                validate(dict(domain,candidates=[a,dict(a,id='duplicate')]))
            with self.assertRaisesRegex(ValueError,'Mixed device'):
                validate(dict(domain,candidates=[a,candidate('other_device',device=0)]))
            with self.assertRaisesRegex(ValueError,'different control'):
                validate(dict(domain,candidates=[a,candidate('other_control',calibration='changed')]))
            with self.assertRaisesRegex(ValueError,'Unadmitted topology'):
                validate(dict(domain,candidates=[dict(a,groups=14)]))
            changed=Path(a['prices']);price=json.loads(changed.read_text())
            price['swiglu_l1_k']=256;changed.write_text(json.dumps(price))
            with self.assertRaisesRegex(ValueError,'L1 implementation'):
                validate(dict(domain,candidates=[a]))
            price['swiglu_l1_k']=128;changed.write_text(json.dumps(price))
            changed=Path(b['prices']);d=json.loads(changed.read_text());d['source_profile_sha256']='changed'
            changed.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'Changed calibration'):validate(domain)

    def test_outer_predicted_cost_changes_returned_native_plan(self):
        a=dict(id='a',status='solved',predicted_forward_ns=10.,output='fixture-plan-a')
        b=dict(id='b',status='solved',predicted_forward_ns=20.,output='fixture-plan-b')
        rejected=dict(id='invalid',status='rejected',predicted_forward_ns=.001,output='not-a-plan')
        self.assertEqual(select([b,rejected,a])['output'],'fixture-plan-a')
        self.assertEqual(select([dict(a,predicted_forward_ns=30.),rejected,b])['output'],'fixture-plan-b')
        for n in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):select([dict(a,predicted_forward_ns=n),b])
        with self.assertRaises(ValueError):select([rejected])


if __name__=='__main__':unittest.main()
