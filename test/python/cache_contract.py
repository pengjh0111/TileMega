import sys, json, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from tilemega.cache import *
from tilemega.build.artifacts import compile_artifact, split_command
from tilemega.fingerprint import calibration_stamps
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
 root=p/'source';body=root/'include/tilemega/Codegen/tasks/PagedGemmTaskBody.h'
 body.parent.mkdir(parents=True);body.write_text('before')
 original=calibration_stamps(root);body.write_text('after')
 changed=calibration_stamps(root)
 assert original['task_bodies'] != changed['task_bodies']
 assert all(original[section] == changed[section] for section in original if section != 'task_bodies')
 # An unchanged generated .cu reuses the same compiled artifact, while a
 # TaskBody in nvcc's transitive -M closure invalidates that artifact alone.
 compiler=p/'fake-nvcc';header=p/'TaskBody.h';header.write_text('first')
 source=p/'plan.cu';source.write_text('#include "TaskBody.h"\n')
 compiler.write_text('#!'+sys.executable+'\n'
  'import pathlib,sys\n'
  'args=sys.argv[1:]\n'
  'if "--version" in args: print("fake nvcc v1"); sys.exit(0)\n'
  'if "-M" in args:\n'
  ' src=next(pathlib.Path(x) for x in args if x.endswith(".cu"))\n'
  ' dependency=next((src.parent/name for name in ("TaskBody.h","OtherBody.h") '
  'if name in src.read_text()),None)\n'
  ' print("tilemega_artifact:",src,dependency)\n'
  ' sys.exit(0)\n'
  'pathlib.Path(args[args.index("-o")+1]).write_bytes(b"compiled")\n')
 compiler.chmod(0o755)
 artifact=p/'artifacts';output=p/'first.so';log=p/'ptxas.txt'
 command=[str(compiler),'-std=c++17','-I',str(p),str(source),'-o',str(output)]
 first=compile_artifact(command,artifact,log)
 assert not first['hit'] and output.read_bytes()==b'compiled'
 second=compile_artifact(command[:-1]+[str(p/'second.so')],artifact,log)
 assert second['hit'] and first['key']==second['key']
 other=p/'OtherBody.h';other.write_text('unchanged')
 independent=p/'other.cu';independent.write_text('#include "OtherBody.h"\n')
 other_command=[str(compiler),'-std=c++17','-I',str(p),str(independent),'-o',str(p/'other.so')]
 other_first=compile_artifact(other_command,artifact,log)
 assert not other_first['hit']
 header.write_text('second')
 third=compile_artifact(command,artifact,log)
 assert not third['hit'] and third['key']!=first['key']
 other_again=compile_artifact(other_command,artifact,log)
 assert other_again['hit'] and other_again['key']==other_first['key']
 assert source.read_text()=='#include "TaskBody.h"\n'
 print('PASS cache records reject changed outputs; TaskBody invalidates only its calibration and nvcc closure; unchanged CUDA sources reuse artifacts')
