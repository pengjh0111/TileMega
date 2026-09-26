from pathlib import Path
import sys,json,shutil,subprocess,argparse,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from tilemega.build.artifacts import compile_artifact
parser=argparse.ArgumentParser();parser.add_argument('--tool',required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
temporary=tempfile.TemporaryDirectory();p=Path(temporary.name)
(p/'a.cu').write_text('#include "input.h"\n__global__ void kernel(int* out) {*out=VALUE;}\n')
(p/'input.h').write_text('#define VALUE 7\n')
nvcc=json.loads(subprocess.check_output([args.tool,'version']))['nvcc']; rows=[]
for name in ['a','b','c']:
 if name!='a':shutil.copy2(p/'a.cu',p/f'{name}.cu')
 if name=='c':(p/'input.h').write_text('#define VALUE 9\n')
 row=compile_artifact([nvcc,'-shared','-Xcompiler=-fPIC','-arch=sm_89',str(p/f'{name}.cu'),'-o',str(p/f'{name}.so')],p/'cache',p/f'{name}.ptxas.txt');rows.append(row)
assert [r['hit'] for r in rows]==[False,True,False]
args.out.write_text(json.dumps(rows,indent=2)+'\n')
print('PASS identical CUDA at a different output path reused; changed header rebuilt')
