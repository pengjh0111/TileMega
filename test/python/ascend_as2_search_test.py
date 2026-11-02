"""CPU-only bounded search guards; synthetic costs never reach a target."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tilemega.bench.as2_models import RUNS
from tilemega.bench.as2_search import select,validate,check_compiler_identity,select_request,evidence


class AscendOuterSearch(unittest.TestCase):
    def test_complete_recipe_costs_prevent_independent_phase_selection(self):
        plain=dict(id='plain',status='solved',predicted_prefill_ns=1.,predicted_decode_ns=10.)
        cached=dict(id='cached',status='solved',predicted_prefill_ns=100.,predicted_decode_ns=5.)
        self.assertEqual(select_request([plain,cached],1024)['id'],'cached')
        self.assertEqual(cached['predicted_request_ns'],100.+1023*5.)
        # A changed measured decode price changes the selected whole recipe;
        # the cheaper plain prefill is never paired with cached decode.
        slower=dict(cached,predicted_decode_ns=20.)
        self.assertEqual(select_request([plain,slower],1024)['id'],'plain')
        with self.assertRaises(ValueError):select_request([plain,cached],16)
        with self.assertRaises(ValueError):select_request([dict(cached,predicted_decode_ns=0)],1024)

    def test_context_provenance_rejects_changed_or_mislabelled_measurements(self):
        with tempfile.TemporaryDirectory(dir=RUNS/'probes') as raw:
            root=Path(raw);cost=root/'cost.json';bundle=root/'contexts.json';price=root/'price.json'
            cost.write_text(json.dumps(dict(status='measured_empirical_context',groups=8,physical_device_id=1)))
            rows=[]
            for i,(low,high,past) in enumerate([(64,127,95),(128,511,319),(512,1086,799)]):
                job=root/str(i);job.mkdir()
                (job/'job.json').write_text(json.dumps(dict(status='passed',executable_sha256='fixture-binary',physical_device_id=1)))
                p=root/(str(i)+'.json');p.write_text(json.dumps(dict(status='measured_body_only',phase='decode',
                    groups=8,physical_device_id=1,job=str(job),binary_sha256='fixture-binary',past=past)))
                rows.append(dict(past_min=low,past_max=high,past=past,source_profile=str(p),
                    source_profile_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            source=dict(schema='tilemega.as2.native_body_contexts.v1',status='measured_body_contexts',phase='decode',
                past_min=64,past_max=1086,forward_count=1023,buckets=rows)
            def save():
                bundle.write_text(json.dumps(source))
                price.write_text(json.dumps(dict(schema='tilemega.as2.native_task_prices.v1',status='sourced_empirical_context',
                    phase='decode',groups=8,past_min=64,past_max=1086,context_aggregation=dict(bucket_counts=[64,384,575]),
                    source_profile=str(bundle),source_profile_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                    source_costs=str(cost),source_costs_sha256=hashlib.sha256(cost.read_bytes()).hexdigest())))
            save();self.assertEqual(evidence(price)[1],1)
            rows[1]['past']=512;save()
            with self.assertRaisesRegex(ValueError,'Context buckets'):evidence(price)
            rows[1]['past']=319;save()
            p=Path(rows[1]['source_profile']);p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Changed context profile'):evidence(price)

    def test_rebuilt_compiler_invalidates_a_frozen_search_identity(self):
        with tempfile.TemporaryDirectory(dir=RUNS/'probes') as raw:
            p=Path(raw)/'compiler';p.write_bytes(b'cpu fixture compiler')
            identity=hashlib.sha256(p.read_bytes()).hexdigest()
            check_compiler_identity(p,identity)
            p.write_bytes(b'rebuilt compiler')
            with self.assertRaisesRegex(ValueError,'compiler changed'):
                check_compiler_identity(p,identity)

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
            price['precision_contract_id']='unimplemented_bf16_streaming';changed.write_text(json.dumps(price))
            with self.assertRaisesRegex(ValueError,'precision contract'):
                validate(dict(domain,candidates=[a]))
            del price['precision_contract_id'];changed.write_text(json.dumps(price))
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
