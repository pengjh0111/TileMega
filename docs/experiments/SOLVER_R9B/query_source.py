#!/usr/bin/env python3
"""Recheck full-kernel occupancy of a fixed-point ablation before measurement."""
import argparse,json,os,pathlib,shutil,sys
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]
sys.path.insert(0,str(E.parent/'SOLVER_V2'))
from measure import run

def query(source,out):
 out.mkdir(parents=True,exist_ok=True);need=8192;free=shutil.disk_usage(out).free//2**20
 (out/'disk.json').write_text(json.dumps(dict(NEED_MIB=need,FREE_MIB=free))+'\n')
 if free<need:raise RuntimeError('insufficient compilation disk')
 target=json.loads((E/'fit/target.json').read_text());sms=target['resources']['num_sms']
 wrapper=out/'query.cu';binary=out/'query'
 wrapper.write_text('#define main tilemega_fixture_main\n#include '+json.dumps(str(source.resolve()))+'\n#undef main\n'+r'''
int main() {
 using namespace tilemega::codegen;
 auto target=tilemega::TargetSpec::Probe();
 if(target.arch_tag!="sm_89" || target.res.num_sms!=SMS) return 3;
 TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(tilemega_l1_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(TaskSmem)));
 TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(tilemega_l2_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(TaskSmem)));
 int l1=target.ActiveBlocksPerSM(reinterpret_cast<void const*>(tilemega_l1_kernel),kHarnessThreads,sizeof(TaskSmem));
 int l2=target.ActiveBlocksPerSM(reinterpret_cast<void const*>(tilemega_l2_kernel),kHarnessThreads,sizeof(TaskSmem));
 std::printf("{\"resident\":%d,\"l1\":%d,\"l2\":%d,\"shared\":%zu}\n",std::min(l1,l2),l1,l2,sizeof(TaskSmem));
}
'''.replace('SMS',str(sms)))
 env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
 cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=sm_89','-lineinfo','-Xptxas=-v','-DTILEMEGA_MIDPOINT_REFINE=0']
 for sub in ('include','third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test'):cmd+=['-I'+str(ROOT/sub)]
 cmd += [wrapper,ROOT/'build-portable/libtilemega.a','-L/usr/local/cuda/lib64','-lcudart','-o',binary]
 if run(cmd,out/'build.log',env):raise RuntimeError('resource query compilation failed')
 if run([binary],out/'query.log',env):raise RuntimeError('resource query failed')
 return json.loads((out/'query.log').read_text())['resident']
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=pathlib.Path);p.add_argument('out',type=pathlib.Path);a=p.parse_args();print(query(a.source,a.out))
