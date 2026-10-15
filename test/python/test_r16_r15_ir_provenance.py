"""Dual frontend/backend provenance cannot silently become a pristine pipeline."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('r15_ir',ROOT/'scripts/r16_r15_ir_provenance.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class R15IRTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  self.old=self.root/'runs/r16/build-r15/tools/tilemega';self.new=self.root/'runs/r16/build-snapshot'/m.FRONTEND_SNAPSHOT/'tools/tilemega'
  for path,data in [(self.old,b'old compiler fixture'),(self.new,b'frontend compiler fixture')]:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  self.patches=[patch.object(m,'R15_COMPILER_SHA',m.sha(self.old)),patch.object(m,'FRONTEND_COMPILER_SHA',m.sha(self.new))]
  for p in self.patches:p.start()
  self.dir=self.root/'runs/r16/r15-ir-fixture';self.dir.mkdir();self.cg=self.dir/'prefill.mlir';self.cg.write_text('unmodified repaired IR fixture');self.export=self.dir/'export.json';self.export.write_text('{}');self.binary=self.dir/'prefill.so'
  for suffix,data in [('',b'actual backend fixture'),('.cu',b'generated actual source'),('.plan.json',b'actual manifest fixture')]:Path(str(self.binary)+suffix).write_bytes(data)
  source=dict(root=str(self.root/'runs/r16/tm-r15'),head=m.R15_HEAD,worktree_diff_sha256=m.hashlib.sha256(b'').hexdigest(),compiler=str(self.old),compiler_sha256=m.sha(self.old),source_files={'include/unchanged.h':'original-body-hash'})
  self.snapshot=Path(str(self.binary)+'.source.json');self.snapshot.write_text(json.dumps(source))
  self.emit=self.dir/'frontend.json';self.build=self.dir/'backend.json'
  front=dict(returncode=0,phase='prefill',batch=1,input_CG_path=str(self.cg),input_CG_sha256=m.sha(self.cg),frontend_source_head=m.FRONTEND_HEAD,frontend_compiler=str(self.new),frontend_compiler_sha256=m.sha(self.new),command=[str(self.new),'compile',str(self.export),'out.cu','--dump-cg',str(self.cg)],input_export=str(self.export),input_export_sha256=m.sha(self.export))
  back=dict(returncode=0,phase='prefill',batch=1,input_CG_path=str(self.cg),input_CG_sha256=m.sha(self.cg),CG_unchanged_after=True,pristine_R15_head=m.R15_HEAD,pristine_R15_compiler_sha256=m.sha(self.old),command=[str(self.old),'compile',str(self.cg),str(self.binary)],source_snapshot_sha256=m.sha(self.snapshot),binary_sha256=m.sha(self.binary),generated_source_sha256=m.sha(str(self.binary)+'.cu'),manifest_sha256=m.sha(str(self.binary)+'.plan.json'))
  self.emit.write_text(json.dumps(front));back.update(frontend_emit_receipt=str(self.emit),frontend_emit_receipt_sha256=m.sha(self.emit));self.build.write_text(json.dumps(back))
  identity=dict(schema=1,source=source,binary_path=str(self.binary),binary_sha256=back['binary_sha256'],generated_source_sha256=back['generated_source_sha256'],manifest_sha256=back['manifest_sha256'],plan=dict(phase='prefill',batch_hi=1));identity['artifact_id']=m.digest(identity);Path(str(self.binary)+'.identity.json').write_text(json.dumps(identity));self.original=copy.deepcopy(identity)
  self.identity=m.attach_frontend_ir(self.binary,self.emit,self.build,self.root)
 def tearDown(self):
  for p in reversed(self.patches):p.stop()
  self.temp.cleanup()
 def test_metadata_attachment_preserves_old_source_and_original_receipt(self):
  self.assertEqual(m.validate_frontend_ir(self.identity,self.root)['label'],'TM-R15-IR')
  stripped=dict(self.identity['source']);del stripped['frontend_ir_provenance'];self.assertEqual(stripped,self.original['source']);self.assertEqual(json.loads(Path(str(self.binary)+'.identity.original.json').read_text()),self.original)
  self.assertEqual(self.identity['artifact_id'],m.digest({k:v for k,v in self.identity.items() if k!='artifact_id'}))
 def test_changed_cg_is_rejected(self):
  self.cg.write_text('modified CG attribute fixture')
  with self.assertRaisesRegex(ValueError,'CG SHA'):m.validate_frontend_ir(self.identity,self.root)
 def test_new_backend_cannot_be_relabelled_as_old(self):
  bad=copy.deepcopy(self.identity);bad['source']['head']='new backend'
  with self.assertRaisesRegex(ValueError,'unchanged R15'):m.validate_frontend_ir(bad,self.root)
 def test_source_files_must_remain_original(self):
  bad=copy.deepcopy(self.identity);bad['source']['source_files']['include/unchanged.h']='altered'
  with self.assertRaisesRegex(ValueError,'source_files'):m.validate_frontend_ir(bad,self.root)
 def test_wrong_frontend_origin_or_compiler_rejected(self):
  bad=copy.deepcopy(self.identity);bad['source']['frontend_ir_provenance']['frontend_source_head']='wrong'
  with self.assertRaisesRegex(ValueError,'origin'):m.validate_frontend_ir(bad,self.root)
  self.new.write_bytes(b'wrong compiler')
  with self.assertRaisesRegex(ValueError,'origin'):m.validate_frontend_ir(self.identity,self.root)
 def test_receipt_or_actual_output_corruption_rejected(self):
  self.emit.write_text('{}')
  with self.assertRaisesRegex(ValueError,'receipt SHA'):m.validate_frontend_ir(self.identity,self.root)
 def test_actual_binary_must_match_official_build(self):
  self.binary.write_bytes(b'changed binary')
  with self.assertRaisesRegex(ValueError,'output SHA'):m.validate_frontend_ir(self.identity,self.root)
 def test_label_and_batch_must_be_explicit(self):
  for field,value in [('label','TM-R15'),('schema',2)]:
   bad=copy.deepcopy(self.identity);bad['source']['frontend_ir_provenance'][field]=value
   with self.assertRaises(ValueError):m.validate_frontend_ir(bad,self.root)
  bad=copy.deepcopy(self.identity);bad['plan']['batch_hi']=16
  with self.assertRaisesRegex(ValueError,'batch'):m.validate_frontend_ir(bad,self.root)
 def test_second_attachment_refused(self):
  with self.assertRaisesRegex(ValueError,'already attached'):m.attach_frontend_ir(self.binary,self.emit,self.build,self.root)

if __name__=='__main__':unittest.main()
