#define main tilemega_fixture_main
#include "/root/TileMega/docs/experiments/SOLVER_R9B/ablations/llama_s4/template/probe/selected.cu"
#undef main

int main() {
 using namespace tilemega::codegen;
 auto target=tilemega::TargetSpec::Probe();
 if(target.arch_tag!="sm_89" || target.res.num_sms!=128) return 3;
 TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(tilemega_l1_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(TaskSmem)));
 TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(tilemega_l2_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(TaskSmem)));
 int l1=target.ActiveBlocksPerSM(reinterpret_cast<void const*>(tilemega_l1_kernel),kHarnessThreads,sizeof(TaskSmem));
 int l2=target.ActiveBlocksPerSM(reinterpret_cast<void const*>(tilemega_l2_kernel),kHarnessThreads,sizeof(TaskSmem));
 std::printf("{\"resident\":%d,\"l1\":%d,\"l2\":%d,\"shared\":%zu}\n",std::min(l1,l2),l1,l2,sizeof(TaskSmem));
}
