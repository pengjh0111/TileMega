# SPDX-License-Identifier: BSD-3-Clause
"""Candidate collection keeps final rounds, binaries and executor evidence distinct."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import shutil
import unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import collect_r16_candidates as collector
import test_r16_native_measurements as native_fixture
from r16_argument_contract import validate_native_arguments
v=collector


class CandidateCollectionTest(unittest.TestCase):
    def setUp(self):
        helper=native_fixture.NativeMeasurementTest();helper.setUp();self.addCleanup(helper.doCleanups)
        first=helper.native_llm_receipt();self.helper=helper;self.fixture=helper.fixture;self.root=helper.root
        value=v.read(first);source=self.root/'frozen';compiler=self.root/'compiler';compiler.write_bytes(b'frozen compiler')
        for phase in ('prefill','decode'):
            binary=Path(value['configuration'][phase+'_binary']);identity=v.read(str(binary)+'.identity.json')
            m=64 if phase=='prefill' else 16
            gemm=dict(index=0,split_k=1,tile_m=m,tile_n=128,tile_k=64,stages=2,impl='mma16')
            identity.update(source={'head':'frozen-head','root':str(source),'compiler_sha256':v.file_sha(compiler)},defines={'TILEMEGA_GEMM_VARIANT_COUNT':'1','TILEMEGA_GEMM_IMPL_0':'0','TILEMEGA_GEMM_TILE_M':str(m),'TILEMEGA_GEMM_TILE_N':'128','TILEMEGA_GEMM_TILE_K':'64','TILEMEGA_GEMM_STAGES':'2'})
            identity['plan']['gemms']=[gemm]
            Path(str(binary)+'.cu').write_text('constexpr GemmRuntimeDesc kRuntimeGemms0[] = {\n{0,1,'+str(m)+',128,64,2},\n};\n')
            Path(str(binary)+'.plan.json').write_text(json.dumps(identity['plan']))
            for suffix,key in [('.cu','generated_source_sha256'),('.plan.json','manifest_sha256')]:identity[key]=v.file_sha(str(binary)+suffix)
            identity['artifact_id']=native_fixture.validation.json_digest({k:x for k,x in identity.items() if k!='artifact_id'})
            Path(str(binary)+'.identity.json').write_text(json.dumps(identity));value['configuration']['identities'][phase]=identity;value['configuration']['plan_manifests'][('prefill','decode').index(phase)]=identity['plan']
        prompt=self.root/'prompt.json';prompt.write_text('[[1,2,3]]');self.script=self.root/'frozen_driver.py'
        self.script.write_text('''import json
from pathlib import Path
from r16_measurement_validation import native_identity

def check_pair(c):
    return {p:native_identity(c[p]) for p in ('prefill','decode')}

def measurement(root,state,name,c):
    d=Path(state[name]['out'])
    if not d.is_absolute():d=root/d
    path=d/'measurements'/('B'+str(c['batch']))/'measurements.json'
    value=json.loads(path.read_text())
    return {k:value[k] for k in ('pid','e2e_seconds','ttft_seconds','tpot_seconds')}
''')
        self.c=dict(id='mma-L1-loop1',model='llama3_2_1b',batch=16,gpu=6,system='TM-seed',mode='L1',loop=1,seed=0,
            snapshot_source=str(source),snapshot_head='frozen-head',compiler=str(compiler),compiler_sha256=v.file_sha(compiler),
            script=str(self.script),script_sha256=v.file_sha(self.script),prompt_ids=str(prompt),config_sha256=value['model_config_sha256'],
            prefill=value['configuration']['prefill_binary'],decode=value['configuration']['decode_binary'])
        self.paths=[];self.jobs=[]
        first_job=next(iter(self.fixture.state[6].values()));self.fixture.state[6]={}
        for rnd in range(3):
            if rnd==0:path=first;row=value;job=first_job
            else:
                path=self.fixture.receipt('solo',6,rnd,1000+rnd*100)
                row=v.read(path);job=self.fixture.state[6].pop(next(n for n in self.fixture.state[6] if f'_r{rnd}_' in n))
                row.update(system=value['system'],model=value['model'],configuration=copy.deepcopy(value['configuration']))
            row['configuration'].update(mode='L1',decode_loop=1);row['prompt_ids_sha256']=v.file_sha(prompt);path.write_text(json.dumps(row));self.paths.append(path)
            actual=dict(driver_sha256=self.c['script_sha256'],calls=[dict(tokens=count,executor='L1',device_loop_used=count>1) for count in (1,1024) for _ in range(4)])
            (path.parent.parent/'actual-execution.json').write_text(json.dumps(actual))
            name=f'rf2_fixture_llama3_2_1b_B16_final{rnd}_{self.c["id"]}';self.jobs.append(name);self.fixture.state[6][name]=job
        self.fixture.persist();self.plan=self.root/'runs/r16/explicit/plan.json';self.plan.parent.mkdir(parents=True)
        self.plan.write_text(json.dumps(dict(root=str(self.root),run_id='fixture',protocol=dict(isl=64,osl=1024,warmup=1,repeats=3,final_fresh_rounds=3),cells=[dict(model=self.c['model'],batch=16,candidates=[self.c])])) )
    def collect(self):return v.collect(self.root,[self.plan])
    def enable_native_alias(self):
        self.script.write_text(self.script.read_text()+'\nfrom r16_argument_contract import validate_native_arguments\n')
        helper=self.script.with_name('r16_argument_contract.py')
        shutil.copyfile(ROOT/'scripts/r16_argument_contract.py',helper)
        self.c.update(system='TM-H3-stage',native_system_label='TM-seed',
            variant={'combine':'stage','cluster_dim':1},script_sha256=v.file_sha(self.script),
            argument_contract_sha256=v.file_sha(helper))
        runner=Path(self.c['snapshot_source'])/'python/tilemega/serving/r16_run.py'
        runner.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'python/tilemega/serving/r16_run.py',runner)
        for rnd,path in enumerate(self.paths):
            row=v.read(path);row['configuration']['runner_sha256']=v.file_sha(runner)
            path.write_text(json.dumps(row));directory=path.parent.parent
            actual=v.read(directory/'actual-execution.json')
            actual['driver_sha256']=self.c['script_sha256']
            (directory/'actual-execution.json').write_text(json.dumps(actual))
            evidence=dict(system=self.c['system'],native_system_label=self.c['native_system_label'],
                variant=self.c['variant'],artifact_ids={p:i['artifact_id']for p,i in row['configuration']['identities'].items()})
            (directory/'candidate-evidence.json').write_text(json.dumps(evidence))
            arguments=['--prefill',self.c['prefill'],'--decode',self.c['decode'],
                '--prompt-ids',self.c['prompt_ids'],'--batch','16','--out',str(directory),
                '--system-label',self.c['native_system_label'],'--mode','L1','--decode-loop','1',
                '--round-index',str(rnd)]
            contract=validate_native_arguments(runner,arguments)
            contract.update(outer_system=self.c['system'],variant=self.c['variant'])
            (directory/'native-argument-contract.json').write_text(json.dumps(contract))
        plan=v.read(self.plan);plan['cells'][0]['candidates']=[self.c]
        self.plan.write_text(json.dumps(plan))

    def test_explicit_native_alias_preserves_outer_candidate_system(self):
        self.enable_native_alias();result=self.collect()
        self.assertFalse(result['rejections']);self.assertEqual(len(result['processes']),3)
        self.assertTrue(result['candidates'][0]['protocol_complete'])
        for row in [*result['processes'],*v.csv_rows(result['candidates'])]:
            self.assertEqual(row['system'],'TM-H3-stage')
            self.assertEqual(row['native_system_label'],'TM-seed')

    def test_alias_cannot_relabel_another_candidate_or_parser_workload(self):
        self.enable_native_alias();self.assertEqual(len(self.collect()['processes']),3)
        directory=self.paths[0].parent.parent
        for damage in ('system','native_system_label','variant','artifact_ids','driver','parser_source','parser_binary'):
            with self.subTest(damage=damage):
                evidence=directory/'candidate-evidence.json';actual=directory/'actual-execution.json'
                contract=directory/'native-argument-contract.json'
                originals={p:p.read_text()for p in (evidence,actual,contract)}
                if damage in ('system','native_system_label','variant','artifact_ids'):
                    row=v.read(evidence)
                    row[damage]={'combine':'cluster'}if damage=='variant'else({'decode':'other-binary'}if damage=='artifact_ids'else'other-candidate')
                    evidence.write_text(json.dumps(row))
                elif damage=='driver':
                    row=v.read(actual);row['driver_sha256']='another-driver';actual.write_text(json.dumps(row))
                else:
                    row=v.read(contract)
                    if damage=='parser_source':row['runner_sha256']='another-parser'
                    else:row['parsed']['decode']='/other/decode.so'
                    contract.write_text(json.dumps(row))
                result=self.collect();self.assertEqual(len(result['processes']),2)
                self.assertNotIn(self.jobs[0],{x['job']for x in result['processes']})
                self.assertTrue(result['rejections'])
                for path,text in originals.items():path.write_text(text)

    def test_arbitrary_native_alias_and_changed_helper_are_rejected(self):
        self.enable_native_alias()
        for label in ('vllm','TM-H1H4'):
            plan=v.read(self.plan);plan['cells'][0]['candidates'][0]['native_system_label']=label
            self.plan.write_text(json.dumps(plan));self.assertFalse(self.collect()['processes'])
        plan=v.read(self.plan);plan['cells'][0]['candidates'][0]['native_system_label']='TM-seed'
        self.plan.write_text(json.dumps(plan))
        self.script.with_name('r16_argument_contract.py').write_text('changed helper')
        result=self.collect();self.assertFalse(result['processes'])
        self.assertTrue(any('argument helper changed'in x['reason']for x in result['rejections']))
    def test_three_real_final_processes_complete_without_default_selection(self):
        result=self.collect();self.assertFalse(result['rejections']);self.assertEqual(len(result['processes']),3)
        row=result['candidates'][0];self.assertEqual(row['status'],'completed');self.assertEqual(row['fresh_processes'],3);self.assertFalse(row['selected_by_report']);self.assertFalse(row['default_promoted']);self.assertFalse(row['ownership_verified'])
        self.assertEqual(v.csv_rows(result['candidates'])[0]['e2e_ms'],10000)
    def test_pilot_only_is_not_final_or_three_rounds(self):
        state=self.fixture.state[6]
        for name in self.jobs:state[name.replace('final','pilot')]=state.pop(name)
        self.fixture.persist();r=self.collect();self.assertFalse(r['processes']);self.assertEqual(r['candidates'][0]['status'],'not_finalized')
    def test_csv_geometry_comes_from_verified_plan_not_implementation_labels(self):
        result=self.collect();row=v.csv_rows(result['candidates'])[0]
        coordinates=json.loads(row['coordinates'])
        for phase,tile_m in [('prefill',64),('decode',16)]:
            self.assertEqual(coordinates[phase]['geometries'],[dict(impl='mma16',tile_m=tile_m,tile_n=128,tile_k=64,stages=2,split_k=1,count=1)])
    def test_failed_guard_short_run_wrong_loop_and_thermal_are_excluded(self):
        for damage in ['failed_job','failed_guard','short','loop','thermal']:
            with self.subTest(damage=damage):
                original=self.paths[0].read_text();path=self.paths[0];directory=path.parents[2];guard=directory/'guard_result.json';original_guard=guard.read_text();actual=path.parent.parent/'actual-execution.json';original_actual=actual.read_text();log=path.parent/'guard.jsonl';original_log=log.read_text()
                if damage=='failed_job':self.fixture.state[6][self.jobs[0]]['status']='failed';self.fixture.persist()
                elif damage=='failed_guard':guard.write_text('{"code":75}')
                elif damage=='short':row=v.read(path);row['max_tokens']=16;path.write_text(json.dumps(row))
                elif damage=='loop':row=v.read(actual);row['calls'][4]['device_loop_used']=False;actual.write_text(json.dumps(row))
                else:log.write_text(original_log.replace('"0x0"','"0x20"'))
                r=self.collect();self.assertEqual(len(r['processes']),2);self.assertEqual(r['candidates'][0]['status'],'partial');self.assertTrue(r['rejections'])
                path.write_text(original);guard.write_text(original_guard);actual.write_text(original_actual);log.write_text(original_log);self.fixture.state[6][self.jobs[0]]['status']='done';self.fixture.persist()
    def test_repeated_pid_and_versions_cannot_form_complete_triplet(self):
        for damage in ['pid','versions']:
            with self.subTest(damage=damage):
                p=self.paths[2];original=p.read_text();row=v.read(p)
                if damage=='pid':row['pid']=v.read(self.paths[0])['pid']
                else:row['versions']['torch']='other-version'
                p.write_text(json.dumps(row));r=self.collect();self.assertEqual(r['candidates'][0]['status'],'rejected_triplet');self.assertFalse(r['candidates'][0]['protocol_complete']);p.write_text(original)
    def test_changed_checkpoint_and_consistently_hashed_false_dispatch_are_rejected(self):
        p=Path(v.read(self.paths[0])['configuration']['checkpoint_manifest']);original=p.read_text();row=v.read(p);row['seed']=9;p.write_text(json.dumps(row));r=self.collect();self.assertFalse(r['processes']);p.write_text(original)
        binary=Path(self.c['decode']);identity=v.read(str(binary)+'.identity.json');identity['plan']['gemms'][0]['impl']='wgmma';Path(str(binary)+'.plan.json').write_text(json.dumps(identity['plan']));identity['manifest_sha256']=v.file_sha(str(binary)+'.plan.json');identity['artifact_id']=native_fixture.validation.json_digest({k:x for k,x in identity.items() if k!='artifact_id'});Path(str(binary)+'.identity.json').write_text(json.dumps(identity))
        for p in self.paths:
            row=v.read(p);row['configuration']['identities']['decode']=identity;row['configuration']['plan_manifests'][1]=identity['plan'];p.write_text(json.dumps(row))
        r=self.collect();self.assertFalse(r['processes']);self.assertTrue(any('actual GEMM implementation' in x['reason'] for x in r['rejections']))
    def test_selection_report_requires_the_same_valid_triplet(self):
        samples=self.collect()['processes'];report=self.plan.parent/'selection-report.json'
        choice=dict(model=self.c['model'],batch=16,status='complete',selected=self.c['id'],candidates=[dict(candidate=self.c['id'],samples=[dict(measurement=x['measurement'],pid=x['pid']) for x in samples])]);report.write_text(json.dumps({'cells':[choice]}));r=self.collect();self.assertTrue(r['candidates'][0]['selected_by_report']);self.assertEqual(len(r['selections']),1);self.assertFalse(r['selections'][0]['default_promoted'])
        choice['candidates'][0]['samples'][0]['pid']+=1;report.write_text(json.dumps({'cells':[choice]}));r=self.collect();self.assertFalse(r['candidates'][0]['selected_by_report']);self.assertTrue(any(x['status']=='unvalidated_selection' for x in r['rejections']))
    def test_explicit_h5_and_h3_paged_companion_interfaces_keep_only_final_commands(self):
        plan=v.read(self.plan);del plan['run_id'];plan['protocol']['fresh_timing_rounds']=plan['protocol'].pop('final_fresh_rounds')
        candidate_file=self.plan.parent/'candidate.json';self.c['file']=str(candidate_file);candidate_file.write_text(json.dumps(self.c));plan['cells'][0]['candidates']=[self.c];plan['cells'][0]['gpu']=6
        jobs=[dict(name=name,gpu=True,timing=True,command=[self.c['script'],'execute','--candidate',str(candidate_file),'--stage','final','--round-index',str(rnd)]) for rnd,name in enumerate(self.jobs)]
        jobs.append(dict(name='not-a-final',gpu=True,timing=True,command=[self.c['script'],'execute','--candidate',str(candidate_file),'--stage','pilot','--round-index','0']))
        plan['cells'][0]['gpu_jobs']=jobs;self.plan.write_text(json.dumps(plan));result=self.collect();self.assertEqual(len(result['processes']),3);self.assertTrue(result['candidates'][0]['protocol_complete'])
        del plan['cells'][0]['gpu_jobs'];self.plan.write_text(json.dumps(plan));(self.plan.parent/'queue_gpu6_preview.json').write_text(json.dumps(jobs));result=self.collect();self.assertEqual(len(result['processes']),3);self.assertTrue(result['candidates'][0]['protocol_complete'])
        c=v.read(candidate_file);c['seed']=9;candidate_file.write_text(json.dumps(c));result=self.collect();self.assertFalse(result['processes']);self.assertIn('changed plan coordinate seed',result['rejections'][0]['reason'])

    def test_mixed_phase_sources_require_exact_complete_bindings(self):
        identities={p:v.read(self.c[p]+'.identity.json') for p in ('prefill','decode')}
        identities['prefill']['source']=dict(identities['prefill']['source'],
            head='older-safe-prefill',root='/frozen/prefill',compiler_sha256='older-cli')
        with self.assertRaises(ValueError):v.validate_phase_sources(self.c,identities)
        candidate=dict(self.c,phase_sources={p:dict(i['source']) for p,i in identities.items()})
        v.validate_phase_sources(candidate,identities)
        for damage in ('missing_phase','missing_compiler','wrong_head','wrong_root','wrong_compiler','wrong_decode'):
            with self.subTest(damage=damage):
                broken=copy.deepcopy(candidate)
                if damage=='missing_phase':del broken['phase_sources']['prefill']
                elif damage=='missing_compiler':del broken['phase_sources']['prefill']['compiler_sha256']
                elif damage=='wrong_decode':broken['snapshot_head']='other-decode'
                else:
                    key={'wrong_head':'head','wrong_root':'root','wrong_compiler':'compiler_sha256'}[damage]
                    broken['phase_sources']['prefill'][key]='other'
                with self.assertRaises(ValueError):v.validate_phase_sources(broken,identities)

    def test_changed_frozen_driver_or_unsupported_interface_is_explicit(self):
        self.script.write_text(self.script.read_text()+'\n# changed\n');r=self.collect();self.assertFalse(r['processes']);self.assertTrue(any('driver changed' in x['reason'] for x in r['rejections']))
        plan=v.read(self.plan);del plan['run_id'];self.plan.write_text(json.dumps(plan));r=self.collect();self.assertFalse(r['candidates']);self.assertEqual(r['rejections'][0]['status'],'unsupported')

if __name__=='__main__':unittest.main()
