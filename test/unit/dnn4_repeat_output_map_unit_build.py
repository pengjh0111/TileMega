"""Build a small GPU layout unit from a certified Repeat artifact; do not run it."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'python'))
from tilemega.build.feature_contract import build_defines
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();O=args.out.resolve();library=args.artifact.resolve()
PYTHON=sys.executable
read=lambda p:json.loads(Path(p).read_text());save=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';O.mkdir(exist_ok=False)
s=Path(str(library)+'.cu').read_text()
profile={k:v for k,v in build_defines(s,[]).items() if k.startswith('TILEMEGA_F3_')}
assert profile.get('TILEMEGA_F3_REPEAT_OUTPUT_MAP')=='1'
model=dict(library=str(library))
count,nmap=map(int,re.search(r'using F3ForwardProgram=Program<16,(\d+),(\d+),',s).groups());stride=count+nmap
raw=re.search(r'inline static constexpr std::array<unsigned,(\d+)> indices=\{\{(.*?)\n  \}\};',s,re.S);tokens=re.findall(r'kAbsent|\d+u',raw[2]);assert len(tokens)==12*stride
virtual={int(i):body for i,body in re.findall(r'struct LoopBind_(\d+) \{(.*?)\n\};',s,re.S)};assert len(virtual)==5
physical={int(i):{k:(None if v=='kAbsent' else int(v[:-1])) for k,v in re.findall(r'(k\w+)=(kAbsent|\d+u)',body)} for i,body in re.findall(r'struct Bind_(\d+) \{(.*?)\n\};',s,re.S)}
mapfields={'kMap','kResidualMap','kQueryMap','kValueMap','kOutputMap'};fields=list(physical[1]);assert len(fields)==23
calls=[];expected=[]
for i in range(5):
 for j,k in enumerate(fields):
  index=i*len(fields)+j
  if k in mapfields:expr=f'auto* p=view.template Map<LoopBind_{i}::{k}>();output[rep*115+{index}]=p?*reinterpret_cast<unsigned const*>(p):~0u;'
  else:expr=f'auto* p=view.template Get<LoopBind_{i}::{k}>();output[rep*115+{index}]=p?unsigned(p->raw()):~0u;'
  calls.append('{'+expr+'}')
for rep in range(12):
 for i in range(5):
  for key in fields:
   value=physical[1+5*rep+i][key];expected.append(0xffffffff if value is None else (0x10000 if key in mapfields else 0x3f00)+value)
assert len(expected)==1380
src=O/'source';src.mkdir();shutil.copytree(R/'include',src/'include')
text='''#include <tilemega/Codegen/tasks/Sm90F3Program.cuh>
#include <vector>
#include <array>
#include <cstring>
#include <stdexcept>
#include <iostream>
using namespace tilemega::codegen::forward_f3;
using E=cutlass::bfloat16_t;
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device{T* data;Device(std::size_t n){Check(cudaMalloc(&data,n*sizeof(T)));}~Device(){cudaFree(data);}void Put(std::vector<T> const& v){Check(cudaMemcpy(data,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}};
'''
text+='\n'.join('struct LoopBind_'+str(i)+' {'+virtual[i]+'\n};' for i in range(5))+'\n'
text+='struct ActualTable {inline static constexpr std::array<unsigned,'+str(len(tokens))+'> indices={{'+','.join(tokens)+'}};};\n'
text+=f'using Base=Bindings<{count},{nmap}>;\n'
text+='''__global__ void Probe(const __grid_constant__ Base binding,unsigned* output){
 unsigned rep=threadIdx.x;if(rep>=12)return;
'''+f'RepeatBindings<Base,{count},{nmap},ActualTable,12> view{{binding,rep}};\n'+'\n'.join(calls)+'\n}\n'
text+='int main(int,char**)try{\n'
text+=f' constexpr unsigned Count={count},Maps={nmap};Base binding{{}};\n'
text+=''' Device<E> values(Count);Device<E*> pointers(Count);Device<unsigned> indices(ActualTable::indices.size()),result(1380);
 std::vector<E> marker(Count);std::vector<E*> address(Count);
 for(unsigned i=0;i<Count;++i){unsigned short bits=0x3f00+i;std::memcpy(&marker[i],&bits,2);address[i]=values.data+i;}
 values.Put(marker);pointers.Put(address);binding.buffers=pointers.data;
 indices.Put(std::vector<unsigned>(ActualTable::indices.begin(),ActualTable::indices.end()));binding.repeat_indices=indices.data;
 for(unsigned i=0;i<Maps;++i){unsigned tag=0x10000+i;std::memcpy(&binding.maps[i],&tag,4);}
'''
text+=' std::vector<unsigned> expected={'+','.join(str(v)+'u' for v in expected)+'};\n'
text+=''' std::vector<unsigned> actual(1380);unsigned long long hash=14695981039346656037ull;
 for(int iteration=0;iteration<3;++iteration){Probe<<<1,32>>>(binding,result.data);Check(cudaGetLastError());Check(cudaDeviceSynchronize());Check(cudaMemcpy(actual.data(),result.data,actual.size()*4,cudaMemcpyDeviceToHost));if(actual!=expected)throw std::runtime_error("Repeat buffer/map marker mismatch");}
 for(unsigned value:actual)for(int byte=0;byte<4;++byte){hash^=(value>>(byte*8))&255;hash*=1099511628211ull;}
 std::cout<<"Repeat optional map1380 PASS\\noutput_hash="<<std::hex<<hash<<"\\n";return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<"\\n";return 1;}
'''
source=src/'unit.cu';source.write_text(text);inputs={str(p):sha(p) for p in src.rglob('*') if p.is_file()}
save(O/'plan.json',dict(evidence='stated beforeCPU/GPU',profile=profile,real_generated_source=model['library']+'.cu',real_generated_source_sha256=sha(model['library']+'.cu'),checks=1380,expected=expected,scope='Exactactual12repetitionx5stagetable/virtualtraitsfromcandidate;115buffer/mapfieldsperrep. GPUdereferencesknownBF16buffermarkersandopaqueCUtensorMapmarkerwords;everyfieldincludingnullcomparedbitexactCPUexpected3launches/process. NoTMAinterpretation/numericalattention/synchronization/speedclaim. Smalllayoutunit17freshGPU0processeswillqualifydynamicnulllookup;wholemodel5andpriorfullbody/ordergatesremainmandatory.'))
binary=O/'dnn4_f3_test';cmd=['taskset','-c','136-143','nvcc','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-Xptxas','-v','-I'+str(src/'include'),'-I'+str(R/'third_party/cutlass/include'),*[f'-D{k}={v}' for k,v in sorted(profile.items())],str(source),'-o',str(binary)]
with (O/'ptxas.log').open('w') as log:subprocess.run([PYTHON,'/root/python/limits.py','--receipt',str(O/'limits.json'),'--max-gib','32','--timeout','1800','--lock','/tmp/tilemega_repeat_output_map_unit.lock','--',*cmd],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',(O/'ptxas.log').read_text())));assert spills and not any(spills)
save(O/'build.json',dict(binary=str(binary),binary_sha256=sha(binary),inputs=inputs,command=cmd,profile=profile,spill_bytes=0,log=str(O/'ptxas.log')));save(O/'result.json',dict(status='prepared',checks=1380,build=str(O/'build.json'),gpu_executed=False));print(dict(prepared=True,checks=1380),flush=True)
