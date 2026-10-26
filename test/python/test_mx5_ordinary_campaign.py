"""Synthetic CPU ordinary-batch reuse; no GPU, compiler, real cost or approval."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
MODULE=Path(os.environ.get('TILEMEGA_ORDINARY_CAMPAIGN_MODULE',ROOT/'python/tilemega/mx5_ordinary_campaign.py'))
if os.environ.get('TILEMEGA_ORDINARY_TEST_CHILD')!='1':
    def test_isolated_CPU_fixture_suite(tmp_path):
        env=dict(os.environ,TILEMEGA_ORDINARY_TEST_CHILD='1',CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
        result=subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).resolve())],env=env,capture_output=True,text=True)
        (tmp_path/'CPU_SYNTHETIC_fixture.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
else:
    spec=importlib.util.spec_from_file_location('tilemega.mx5_ordinary_campaign',MODULE)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    spec=importlib.util.spec_from_file_location('ordinary_ledger_CPU_fixture',ROOT/'test/python/test_mx5_resolved_top4.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    x=f.m.exception
    def save(path,value):return f.save(path,value)

    class Batch:
        def __init__(self,root,harness,action,kind='gemm'):
            self.root=root;root.mkdir();self.h=harness
            original=harness.add_variant(action);self.catalog=m._load(m.record(original));base=copy.deepcopy(self.catalog['rows'][-1]['row']);assert base['kind']==kind
            self.base=base;self.rows=[];self.keys=[];self.checked=[];self.references=[];self.dir=root/'batch';self.dir.mkdir();self.source_root=root/'retained_source';self.source_root.mkdir()
            archive=m.check_file(harness.sources[0]['archive'])
            import tarfile
            with tarfile.open(archive,'r:gz')as tar:
                for item in tar.getmembers():
                    if item.isfile():
                        p=self.source_root/item.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(item).read())
            for name in ['experiments/mx5/campaign.py','experiments/mx5/stage_bench.py']:
                p=self.source_root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
            self.runner=self.source_root/'experiments/mx5/campaign.py';self.bench=self.source_root/'experiments/mx5/stage_bench.py'
            for index in range(24):
                row=copy.deepcopy(base);row['key']['n']+=index*16;row['r_next_reference']['key']['n']=row['key']['n']
                raw=copy.deepcopy(row['rounds'][0]['measurement']);raw['key']['n']=row['key']['n'];self.keys.append(raw['key'])
                parent=dict(kind=kind,key=row['key'],library=raw['key']['library'],provenance=row['provenance'],resources=row['resources'],r_next_reference=row['r_next_reference'],fits=True,query_only=True,query_status=0,iterations=0,selected_resource_step=0,uses=[dict(model='CPU_SYNTHETIC',stage='qkv')])
                self.checked.append(parent);self.references.append(row['r_next_reference'])
                self.rows.append(row)
            self.keys_ref=save(self.dir/'keys.json',self.keys);self.checked_path=root/'selected.jsonl'
            self.checked_path.write_text(''.join(json.dumps(r)+'\n'for r in self.checked));(self.dir/'checked.jsonl').write_bytes(self.checked_path.read_bytes())
            save(self.dir/'references.json',dict(references=self.references))
            self.job=dict(kind=kind,keys=str(self.dir/'keys.json'),directory=str(self.dir),keys_sha256=self.keys_ref['sha256'],rows=24,suggested_card=7)
            self.manifest=save(root/'campaign.json',dict(schema_version=1,checked=str(self.checked_path),checked_sha256=m.record(self.checked_path)['sha256'],processes_per_key=3,rows=24,jobs=[self.job]))
            for number in [1,2,3]:
                raw=[]
                for index,row in enumerate(self.rows):
                    v=copy.deepcopy(row['rounds'][number-1]['measurement']);v['key']['n']=row['key']['n'];raw.append(v)
                path=self.dir/f'round{number}.jsonl';path.write_text(''.join(json.dumps(r)+'\n'for r in raw));path.with_suffix('.log').write_text('CPU_SYNTHETIC not an executed benchmark\n')
                command=[sys.executable,str(self.bench),'--keys',str(self.dir/'keys.json'),'--out',str(path),'--round',str(number),'--iterations','3']
                save(self.dir/f'process{number}.json',dict(round=number,command=command,returncode=0,CPU_SYNTHETIC=True))
            save(self.dir/'guard/guard_result.json',dict(code=0,CPU_SYNTHETIC=True));(self.dir/'guard/guard.jsonl').write_text(json.dumps(dict(phase='after',exit_code=0,CPU_SYNTHETIC=True))+'\n')
            save(self.dir/'scheduler_job.json',dict(card=7,command=['CPU_SYNTHETIC_GUARD','7','900',str(self.dir/'guard')],keys_sha256=self.keys_ref['sha256']))
            self.snapshot=save(root/'snapshot.json',dict(root=str(self.source_root),files={str(p.relative_to(self.source_root)):m.record(p)['sha256']for p in self.source_root.rglob('*')if p.is_file()},CPU_SYNTHETIC=True))
            binary=Path(self.keys[0]['library']);key=self.base['key'];command=[str(harness.compiler),'-std=c++17','-O3','-shared','-Xcompiler','-fPIC,-resource-usage','-fno-strict-aliasing','-DTILEMEGA_SYNC_V3=1','-DTILEMEGA_WATCHDOG=0','-DMX5_TEST_THREADS='+str(key['threads']),'-DMX5_TEST_TN='+str(key['tn']),'-DMX5_TEST_DEPTH='+str(key['depth']),'-DMX5_BENCH_PW='+str(key['prefetch']),'-Iinclude','experiments/mx5/stage_bench.cu','-o',str(binary)]
            meta=save(root/'library.json',dict(body=kind,bench=True,returncode=0,source_sha256=base['provenance']['source_sha256'],binary_sha256=base['provenance']['binary_sha256'],base_head=base['provenance']['base_head'],command=command))
            query=save(root/'CPU_query_witness.json',dict(CPU_SYNTHETIC=True))
            self.admission=save(root/'ADMISSION.json',dict(campaign=self.manifest,fresh_processes_per_batch=3,iterations_minimum=3,rotation_bytes_minimum=256*1024**2,selected=m.record(self.checked_path),source_sha256=harness.stage_sha,source_snapshot=self.snapshot,actual_checked=query,actual_raw=query,actual_query_summary=query,tools=[m.record(self.runner),m.record(self.bench)],libraries=[dict(binary=m.record(binary),metadata=meta,source_archive=harness.sources[0]['archive'])],CPU_SYNTHETIC=True))
            save(self.dir/'ROOT_RUN_CONTEXT.json',dict(admission=self.admission,manifest=self.manifest,job=0,iterations_minimum=3,child_PYTHONPATH=str(self.source_root/'python'),CPU_SYNTHETIC=True))
            self.parent=root/'qualified_parent.json';self.make_parent()

        def make_parent(self):
            q=m._q();result=q.table.aggregate([self.dir/f'round{n}.jsonl'for n in [1,2,3]],[self.dir/'references.json'],card='C550 GPU 7');parents={q.table.key_id(r['kind'],r['key']):r for r in self.checked}
            for row in result['rows']:
                c=parents[q.table.key_id(row['kind'],row['key'])];row.update(uses=c['uses'],resource_selection={'selected_resource_step':0},card='7',campaign_batch=dict(manifest=self.manifest['path'],job=0))
            save(self.parent,result)

        def capture(self):return m.capture(self.manifest['path'],0,self.admission['path'],self.parent,self.base['kind'],self.base['key'])
        def ordinary_catalog(self):
            view=self.capture();entry=dict(view,authorization=self.catalog['rows'][-1]['authorization']);doc=copy.deepcopy(self.catalog);doc['rows']=[entry];path=self.root/'ordinary_catalog.json';save(path,doc);return path,view
        def mutate_json(self,path,change):v=json.loads(path.read_text());change(v);save(path,v)

    def case(tmp_path,monkeypatch):
        h=f.Fixture(tmp_path,monkeypatch,threads=512);parent=h.h.artifact(mt=256,frame=8,query=False);next_=x.next_width_diagnostic(h.h.origin,0,[parent]);assert next_.action==x.Action.NEXT_WIDTH
        child=h.h.artifact(next_.data['next_pin'],ordinal=1);action=type('CPUAction',(),dict(details=x.legacy.FrozenJSON.freeze(dict(authorization=next_.data['authorization']))))()
        batch=Batch(tmp_path/'ordinary',h.h.h,action);return h,parent,child,batch

    def test_ordinary_24key_no_PID_reuse_binds_512_costs_and_keeps_parent(tmp_path,monkeypatch):
        h,parent,child,b=case(tmp_path,monkeypatch);path,view=b.ordinary_catalog();before=b.parent.read_bytes();m.verify(view,view['row'])
        assert len(m._load(view['ordinary_campaign']['witnesses']['keys']))==24
        assert all('pid'not in m._load(p)for p in view['ordinary_campaign']['witnesses']['processes'])
        assert 'resource_selection'not in view['row'] and 'resource_selection'in json.loads(before)['rows'][0]
        result=x.bind_descendant_costs(h.h.origin,0,[parent,child],[m.record(path)])
        assert result.action==x.Action.COST_BOUND,result.as_dict();assert b.parent.read_bytes()==before
        assert result.data['prediction']['selection_anchor']==0

    def test_same_ordinary_helper_used_by_true576_ledger_replay(tmp_path,monkeypatch):
        h=f.Fixture(tmp_path,monkeypatch,threads=576);ledger=x.legacy.PlanLedger.start(h.h.origin,0);bundle=h.h.artifact(mt=228,query=False);node=m._load(bundle)
        from tilemega.mx5_plan_receipt import assemble
        scoped=assemble(node['native_receipt']['path'],ledger,stage_sources=h.h.h.sources,resources=node['resources']['path']);p=tmp_path/'attempt0.json';save(p,scoped);ledger=ledger.append_attempt(p)
        action=x.legacy.next_plan_retry(ledger);assert action.kind==x.legacy.ActionKind.NEEDS_STAGE_MEASUREMENTS
        b=Batch(tmp_path/'ordinary',h.h.h,action);path,_=b.ordinary_catalog();catalog=x.legacy.VariantCatalog.load(path);ready=x.legacy.next_plan_retry(ledger,catalog);assert ready.kind==x.legacy.ActionKind.BUILD_REDUCED_PLAN
        ledger=ledger.commit_retry(ready,catalog);bundle=h.h.artifact(ledger.pin.data,ordinal=1,mt=220);node=m._load(bundle)
        scoped=assemble(node['native_receipt']['path'],ledger,stage_sources=h.h.h.sources,resources=node['resources']['path'],occupancy=node['query']['path']);p=tmp_path/'attempt1.json';save(p,scoped);ledger=ledger.append_attempt(p)
        out=tmp_path/'ledger.json';ledger.write(out);replayed,_=f.m._replay_576(m.record(out),h.h.origin,0)
        assert replayed.sha256==ledger.sha256 and len(replayed.authorizations)==1

    @pytest.mark.parametrize('problem',['process_rc','round','wrong_argv','raw_other_key','raw_median','guard','admission','source','tool','parent_other_row','pid_duplicate','key_projection','mode','resource_selection','authorization'])
    def test_malformed_original_evidence_or_catalog_blocks_without_prediction(tmp_path,monkeypatch,problem):
        h,parent,child,b=case(tmp_path,monkeypatch);view=b.capture()
        if problem=='process_rc':b.mutate_json(b.dir/'process2.json',lambda v:v.update(returncode=1))
        elif problem=='round':b.mutate_json(b.dir/'process2.json',lambda v:v.update(round=1))
        elif problem=='wrong_argv':b.mutate_json(b.dir/'process2.json',lambda v:v['command'].__setitem__(v['command'].index('--out')+1,'/wrong/raw.jsonl'))
        elif problem in ('raw_other_key','raw_median'):
            p=b.dir/'round2.jsonl';data=[json.loads(l)for l in p.read_text().splitlines()];data[-1]['key']['n']+=16 if problem=='raw_other_key' else 0
            if problem=='raw_median':data[-1]['median_ns']+=1
            p.write_text(''.join(json.dumps(r)+'\n'for r in data))
            with pytest.raises(ValueError):b.capture()
            return
        elif problem=='guard':b.mutate_json(b.dir/'guard/guard_result.json',lambda v:v.update(code=1))
        elif problem=='admission':b.mutate_json(Path(b.admission['path']),lambda v:v.update(source_sha256='0'*64))
        elif problem=='source':(b.source_root/'experiments/mx5/stage_bench.py').write_text('changed CPU source')
        elif problem=='tool':view['ordinary_campaign']['tools'][0]['sha256']='0'*64
        elif problem=='parent_other_row':
            b.mutate_json(b.parent,lambda v:v['rows'][-1].update(median_ns=1))
            with pytest.raises(ValueError,match='annotated parent'):b.capture()
            return
        elif problem=='pid_duplicate':
            for n in[1,2,3]:b.mutate_json(b.dir/f'process{n}.json',lambda v:v.update(pid=123))
            with pytest.raises(ValueError,match='PIDs'):b.capture()
            return
        elif problem=='key_projection':view['keys']=save(tmp_path/'one_key.json',[b.keys[0]])
        elif problem=='mode':view['evidence_mode']='unknown'
        elif problem=='resource_selection':view['row']['resource_selection']={'selected_resource_step':1};view['row_sha256']=x.legacy.digest(view['row'])
        else:
            doc=copy.deepcopy(b.catalog);entry=dict(view,authorization=copy.deepcopy(doc['rows'][-1]['authorization']));entry['authorization']['candidate_id']=1;doc['rows']=[entry];path=tmp_path/'wrongauth.json';save(path,doc)
            result=x.bind_descendant_costs(h.h.origin,0,[parent,child],[m.record(path)]);assert result.action==x.Action.NEEDS_COSTS and result.data['prediction'] is None;return
        if problem=='resource_selection':
            doc=copy.deepcopy(b.catalog);doc['rows']=[dict(view,authorization=doc['rows'][-1]['authorization'])];path=tmp_path/'fake_selection.json';save(path,doc)
            with pytest.raises(ValueError):x.legacy.VariantCatalog.load(path)
        else:
            with pytest.raises((ValueError,KeyError,TypeError)):x._fresh(dict(view,qualification_tool=m.record(ROOT/'experiments/mx5/stage_table.py')),view['row'])

    def test_current_PID_mode_duplicate_PID_remains_rejected(tmp_path,monkeypatch):
        h=f.f.Fixture(tmp_path,monkeypatch);parent=h.artifact(mt=256,frame=8,query=False);next_=x.next_width_diagnostic(h.origin,0,[parent]);catalog=h.catalog(next_.data['authorization']);doc=m._load(catalog);entry=doc['rows'][0]
        process=m._load(entry['fresh_processes'][0]);p=tmp_path/'duplicate_PID.json';save(p,dict(m._load(entry['fresh_processes'][1]),pid=process['pid']));entry['fresh_processes'][1]=m.record(p)
        with pytest.raises(ValueError,match='PIDs'):x._fresh(dict(entry,qualification_tool=doc['qualification_tool']),entry['row'])
