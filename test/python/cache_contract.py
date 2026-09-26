import sys, json, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from tilemega.cache import *
from tilemega.build.artifacts import split_command
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'output').write_text('a');record_outputs(p/'record.json',[p/'output']);assert valid_record(p/'record.json')
 (p/'output').write_text('b');assert not valid_record(p/'record.json')
 model=p/'config.json';model.write_text('{}');exporter=p/'export.py';exporter.write_text('export')
 e=export_key(model,exporter,'decode',1,1088)
 a=plan_key(e,{'body':'one'},'source',{}, {},1,(64,1086))
 b=plan_key(e,{'body':'two'},'source',{}, {},1,(64,1086))
 assert a!=b and e==export_key(model,exporter,'decode',1,1088)
 out,sources,opts=split_command(['nvcc','-std=c++17','-x','cu','a.cu','-x','cu','b.cpp','-shared','-I','/dir with spaces','-L/lib','-lcudart','-o','out.so'])
 assert len(sources)==2 and opts==['-std=c++17','-I','/dir with spaces']
 print('PASS cache records reject changed outputs; target stamps invalidate only plan; nvcc inputs preserve include arguments')
