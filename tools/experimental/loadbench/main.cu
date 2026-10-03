// SPDX-License-Identifier: BSD-3-Clause
#include "kernels.cuh"
#include <tilemega/Codegen/executor/ServingLaunch.cuh>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>
using namespace loadbench;
namespace {
constexpr std::size_t MiB=1ull<<20;
void Check(cudaError_t value) {if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
double Median(std::vector<double> v) {std::sort(v.begin(),v.end());return (v[v.size()/2]+v[(v.size()-1)/2])/2;}
std::string Quote(std::string const& text) {
  std::ostringstream out;out<<'"';for(unsigned char c:text) {
    if(c=='"'||c=='\\')out<<'\\'<<c;
    else if(c=='\n')out<<"\\n";
    else if(c=='\r')out<<"\\r";
    else if(c=='\t')out<<"\\t";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
    else out<<c;
  }out<<'"';return out.str();
}
std::string Capture(char const* command) {
  FILE* pipe=popen(command,"r");if(!pipe)return "query failed";
  std::string text;char buffer[2048];while(fgets(buffer,sizeof(buffer),pipe))text+=buffer;
  pclose(pipe);return text;
}
struct Snapshot {std::string clocks,apps;bool external=false;};
Snapshot Observe() {
  Snapshot s;
  s.clocks=Capture("nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>&1");
  s.apps=Capture("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>&1");
  std::istringstream lines(s.apps);std::string line;
  while(std::getline(lines,line)) {std::istringstream row(line);int pid;if(row>>pid)if(pid!=getpid())s.external=true;}
  return s;
}
struct Buffer {
  char* p=nullptr;std::size_t bytes;
  explicit Buffer(std::size_t n):bytes(n){Check(cudaMalloc(&p,n));Check(cudaMemset(p,0,n));}
  ~Buffer(){if(p)cudaFree(p);}
  Buffer(Buffer const&)=delete;Buffer& operator=(Buffer const&)=delete;
};
struct Stat {double ms;std::vector<double> samples;};
struct Bench {
  cudaDeviceProp device{};tilemega::arch::RuntimeCaps caps;
  Buffer flush{256*MiB},sink{MiB},counters{MiB},times{MiB};
  cudaEvent_t start,stop;std::vector<std::string> points;int repeats=10;
  Bench():caps(tilemega::arch::RuntimeCapsFor<void>()) {
    Check(cudaGetDeviceProperties(&device,0));
    caps=tilemega::arch::RuntimeCapsForTag("sm_"+std::to_string(device.major*10+device.minor));
    Check(cudaEventCreate(&start));Check(cudaEventCreate(&stop));
    Fill<<<device.multiProcessorCount,256>>>(flush.p,flush.bytes);Check(cudaDeviceSynchronize());
  }
  ~Bench(){cudaEventDestroy(start);cudaEventDestroy(stop);}
  unsigned* Sink(){return reinterpret_cast<unsigned*>(sink.p);}
  unsigned long long* Counters(){return reinterpret_cast<unsigned long long*>(counters.p);}
  unsigned long long* Times(){return reinterpret_cast<unsigned long long*>(times.p);}
  Stat Time(std::function<void()> const& launch,std::function<void()> const& after={}) {
    std::vector<double> values;
    for(int r=0;r<repeats;++r) {
      StreamReadKernel<<<device.multiProcessorCount*4,256>>>(reinterpret_cast<float4*>(flush.p),flush.bytes/16,1,977,reinterpret_cast<float*>(sink.p));
      Check(cudaDeviceSynchronize());Check(cudaMemset(counters.p,0,counters.bytes));
      Check(cudaEventRecord(start));launch();Check(cudaGetLastError());
      Check(cudaEventRecord(stop));Check(cudaEventSynchronize(stop));float ms;
      Check(cudaEventElapsedTime(&ms,start,stop));values.push_back(ms);
      if(after)after();
    }return {Median(values),values};
  }
  void Add(std::string suite,std::string fields,Stat const& stat,double moved,double value=-1,std::string metric="gbps",std::vector<double> extra={}) {
    std::ostringstream out;out<<std::setprecision(12)<<"{\"suite\":"<<Quote(suite)<<','<<fields
      <<",\"median_ms\":"<<stat.ms<<",\"bytes\":"<<moved<<",\""<<metric<<"\":"<<(value==-1?moved/stat.ms/1e6:value)<<",\"samples_ms\":[";
    for(std::size_t i=0;i<stat.samples.size();++i){if(i)out<<',';out<<stat.samples[i];}
    out<<"],\"metric_samples\":[";
    for(std::size_t i=0;i<extra.size();++i){if(i)out<<',';out<<extra[i];}
    out<<"]}";points.push_back(out.str());
    std::cerr<<suite<<' '<<fields<<" median_ms="<<stat.ms<<'\n';
  }
  void Skip(std::string suite,std::string reason) {
    points.push_back("{\"suite\":"+Quote(suite)+",\"status\":\"unsupported\",\"reason\":"+Quote(reason)+"}");
  }
};
void CpLaunch(int depth,int policy,char const* p,std::size_t count,int passes,int grid,int threads) {
  auto bytes=std::size_t(depth)*threads*16;
#define CP(D,P) CpStream<D,P><<<grid,threads,bytes>>>(p,count,passes)
  if(depth==4){switch(policy){case 0:CP(4,0);break;case 1:CP(4,1);break;case 2:CP(4,2);break;case 3:CP(4,3);break;}}
  else{switch(policy){case 0:CP(8,0);break;case 1:CP(8,1);break;case 2:CP(8,2);break;case 3:CP(8,3);break;}}
#undef CP
}
void Stream(Bench& b,int method,int param,char const* p,std::size_t bytes,int passes,int grid,int threads) {
  switch(method) {
  case 1:StreamReadKernel<<<grid,threads>>>(reinterpret_cast<float4 const*>(p),bytes/16,passes,977,reinterpret_cast<float*>(b.sink.p));break;
  case 2:if(param==4)NcStream<4><<<grid,threads>>>(p,bytes/16,passes,b.Sink());else NcStream<8><<<grid,threads>>>(p,bytes/16,passes,b.Sink());break;
  case 3:CpLaunch(param,0,p,bytes/16,passes,grid,threads);break;
  case 4:CpLaunch(param>=4096?4:8,param%4096==128?1:2,p,bytes/16,passes,grid,threads);break;
  case 5:CpLaunch(param==4?4:8,3,p,bytes/16,passes,grid,threads);break;
  default:throw std::invalid_argument("unknown method");
  }
}
struct Method {int method,param,grid,threads;};
int CalibGrid(Bench& b) {
  int resident=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,StreamReadKernel,256,0));
  return b.device.multiProcessorCount*std::max(1,resident);
}
std::string MethodFields(Method m,std::size_t bytes,char alloc,int curve=0) {
  return "\"method\":"+std::to_string(m.method)+",\"param\":"+std::to_string(m.param)+",\"grid\":"+std::to_string(m.grid)+",\"threads\":"+std::to_string(m.threads)+",\"working_set_bytes\":"+std::to_string(bytes)+",\"allocation\":"+Quote(std::string(1,alloc))+",\"curve\":"+std::to_string(curve);
}
Stat CeilingPoint(Bench& b,Method m,std::size_t bytes,char alloc,Buffer& reused) {
  int passes=std::max(2,int(2.0e9/double(bytes)));
  if(alloc=='A')return b.Time([&]{Stream(b,m.method,m.param,reused.p,bytes,passes,m.grid,m.threads);});
  if(alloc=='B'){Buffer data(bytes);return b.Time([&]{Stream(b,m.method,m.param,data.p,bytes,passes,m.grid,m.threads);});}
  Buffer p0(512*MiB),p1(512*MiB),p2(512*MiB),p3(512*MiB);char* pointers[]={p0.p,p1.p,p2.p,p3.p};
  return b.Time([&]{for(int pass=0;pass<passes;++pass)for(int i=0;i<4;++i) {
    auto n=std::min(512*MiB,bytes>i*512*MiB?bytes-i*512*MiB:0);
    if(n)Stream(b,m.method,m.param,pointers[i],n,1,m.grid,m.threads);
  }});
}
void Ceiling(Bench& b,bool replay,Method specified) {
  Buffer reused(2048*MiB);Fill<<<b.device.multiProcessorCount*4,256>>>(reused.p,reused.bytes);Check(cudaDeviceSynchronize());
  int sizes[]={256,512,1024,1536,2048};Method baseline{1,0,CalibGrid(b),256};Method best=specified;double peak=0;
  if(!replay) {
    for(int factor:{1,2,4})for(int threads:{128,256})for(auto spec:std::vector<std::pair<int,int>>{{1,0},{2,4},{2,8},{3,4},{3,8},{4,128},{4,256},{4,4096+128},{4,4096+256},{5,0},{5,4}}) {
      Method method{spec.first,spec.second,b.device.multiProcessorCount*factor,threads};
      for(int mib:sizes){auto bytes=mib*MiB;auto stat=CeilingPoint(b,method,bytes,'A',reused);auto moved=bytes*std::max(2,int(2.0e9/bytes));
        b.Add("MB-1a",MethodFields(method,bytes,'A'),stat,moved);
        if(moved/stat.ms/1e6>peak){peak=moved/stat.ms/1e6;best=method;}
      }
    }
  }
  for(char alloc:{'A','B','C'})for(Method method:{baseline,best})for(int mib:sizes) {
    auto bytes=mib*MiB;auto stat=CeilingPoint(b,method,bytes,alloc,reused);
    b.Add("MB-1a",MethodFields(method,bytes,alloc),stat,bytes*std::max(2,int(2.0e9/bytes)));
  }
  // Reproduce F14's separation with the same allocation and occupancy.
  for(int curve=1;curve<=2;++curve) {
    if(curve==2)std::this_thread::sleep_for(std::chrono::seconds(10));
    for(int mib:sizes){auto bytes=mib*MiB;auto stat=CeilingPoint(b,baseline,bytes,'A',reused);
      b.Add("MB-1a-reproduction",MethodFields(baseline,bytes,'A',curve),stat,bytes*std::max(2,int(2.0e9/bytes)));}
  }
}
int ResidentShared(Bench const& b);
void Pages(Bench& b) {
  int grid=b.device.multiProcessorCount;
  {
    Buffer data(512*MiB);int pages=data.bytes/(grid*16384);auto bytes=std::size_t(grid)*pages*16384;
    Check(cudaFuncSetAttribute(LoaderAnchor,cudaFuncAttributeMaxDynamicSharedMemorySize,1024+5*16384));
    b.Add("MB-1b-anchor","\"loader_warps\":1,\"page_bytes\":16384,\"pages\":5,\"grid\":"+std::to_string(grid),b.Time([&]{LoaderAnchor<<<grid,160,1024+5*16384>>>(data.p,pages);}),bytes);
  }
  Buffer data(2048*MiB);
  for(int page_bytes:{8192,16384})for(int pages:{3,4,5,6}) {
    int pool=1024+page_bytes*pages;
    // A grid with SM-count blocks alone does not prove one CTA per SM.
    // Reserve unused shared memory so even the smallest pool is resident once.
    int shared=std::max(pool,ResidentShared(b));if(shared>b.device.sharedMemPerBlockOptin)continue;
    Check(cudaFuncSetAttribute(PageLoader,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    int resident=0;
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,PageLoader,256,shared));
    if(resident!=1){b.Skip("MB-1b","one resident CTA per SM could not be enforced");continue;}
    for(int warps:{1,2,4})for(bool consume:{false,true}) {
      int per_cta=data.bytes/(grid*page_bytes);double bytes=double(per_cta)*grid*page_bytes;
      auto fields="\"reserved_smem_bytes\":"+std::to_string(shared)+",\"resident_limit\":1,\"loader_warps\":"+std::to_string(warps)+",\"page_bytes\":"+std::to_string(page_bytes)+",\"pages\":"+std::to_string(pages)+",\"consume\":"+(consume?"true":"false")+",\"bulk\":false";
      b.Add("MB-1b",fields,b.Time([&]{PageLoader<<<grid,128+32*warps,shared>>>(data.p,per_cta,page_bytes,pages,warps,consume,false,b.Sink());}),bytes);
    }
    if(b.caps.bulk_copy)for(bool consume:{false,true}) {
      int per_cta=data.bytes/(grid*page_bytes);
      b.Add("MB-1b","\"loader_warps\":1,\"page_bytes\":"+std::to_string(page_bytes)+",\"pages\":"+std::to_string(pages)+",\"consume\":"+(consume?"true":"false")+",\"bulk\":true",b.Time([&]{PageLoader<<<grid,160,shared>>>(data.p,per_cta,page_bytes,pages,1,consume,true,b.Sink());}),double(per_cta)*grid*page_bytes);
    }
  }
  if(!b.caps.bulk_copy)b.Skip("MB-1b-bulk","arch::Caps::kBulkCopy is false");
}
template<int TN,int TK,int S>
void Shape(Bench& b,char const* data,std::size_t bytes,int k,int grid,bool row,int method) {
  int shared=TN*TK*2*S;std::size_t tiles=bytes/(TN*k*2);
#define SH(M) do {Check(cudaFuncSetAttribute(GemvStream<TN,TK,S,M>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));GemvStream<TN,TK,S,M><<<grid,128,shared>>>(data,tiles,k,row,b.Sink());}while(0)
  switch(method){case 0:SH(0);break;case 1:SH(1);break;case 2:SH(2);break;case 3:SH(3);break;case 4:SH(4);break;case 5:SH(5);break;}
#undef SH
}
void Shapes(Bench& b) {
  Buffer data(2048*MiB);
  for(auto geometry:std::vector<std::tuple<int,int,int>>{{32,128,4},{32,128,7},{128,64,4}}) {
    int tn=std::get<0>(geometry),tk=std::get<1>(geometry),s=std::get<2>(geometry);if(tn*tk*2*s>b.device.sharedMemPerBlockOptin)continue;
    for(int k:{2048,8192})for(int active:{50,75,100})for(bool row:{false,true})for(int method=0;method<6;++method) {
      int grid=std::max(1,b.device.multiProcessorCount*active/100);
      auto stat=b.Time([&]{if(tn==128)Shape<128,64,4>(b,data.p,data.bytes,k,grid,row,method);else if(s==4)Shape<32,128,4>(b,data.p,data.bytes,k,grid,row,method);else Shape<32,128,7>(b,data.p,data.bytes,k,grid,row,method);});
      b.Add("MB-1c","\"tile_n\":"+std::to_string(tn)+",\"tile_k\":"+std::to_string(tk)+",\"stages\":"+std::to_string(s)+",\"K\":"+std::to_string(k)+",\"active_pct\":"+std::to_string(active)+",\"row\":"+(row?"true":"false")+",\"method\":"+std::to_string(method),stat,data.bytes);
    }
  }
}
int ResidentShared(Bench const& b){return std::min(int(b.device.sharedMemPerBlockOptin),int(b.device.sharedMemPerMultiprocessor/2+256));}
double HopLatency(Bench& b,int foreground) {
  std::vector<unsigned long long> times(foreground*2);
  Check(cudaMemcpy(times.data(),b.times.p,times.size()*8,cudaMemcpyDeviceToHost));
  unsigned long long last=0,first=~0ull;
  for(int i=0;i<foreground;++i){last=std::max(last,times[2*i]);first=std::min(first,times[2*i+1]);}
  return double(first)-double(last);
}
void Hops(Bench& b) {
  int grid=b.device.multiProcessorCount;Buffer data(std::size_t(grid)*512*1024);
  int shared=ResidentShared(b);Check(cudaFuncSetAttribute(Hop,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  for(int kib:{64,512})for(bool background:{false,true})for(auto policy:std::vector<std::pair<int,int>>{{0,1},{1,1},{1,4}}) {
    std::vector<double> latency;
    auto stat=b.Time([&]{Hop<<<grid,128,shared>>>(data.p,kib*1024,background,policy.first,policy.second,b.Counters(),b.Times(),b.Sink());},
                    [&]{latency.push_back(HopLatency(b,background?grid/2:grid));});
    b.Add("MB-1d","\"KiB\":"+std::to_string(kib)+",\"background\":"+(background?"true":"false")+",\"task_events\":"+(policy.first?"true":"false")+",\"kappa\":"+std::to_string(policy.second),stat,double(kib)*1024*(background?grid/2:grid),Median(latency),"hop_ns",latency);
  }
  if(b.caps.cluster)for(int cluster:{2,4,8})for(int kib:{64,512})for(bool background:{false,true}) {
    int foreground=background?grid/2:grid;
    if(cluster>b.caps.max_cluster_size || grid%cluster || foreground%cluster) {b.Skip("MB-1d-cluster","cluster size does not divide the foreground grid or exceeds caps");continue;}
    Check(cudaFuncSetAttribute(ClusterHop,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=cluster;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;
    cudaLaunchConfig_t config{};config.gridDim=dim3(grid);config.blockDim=dim3(128);config.dynamicSmemBytes=shared;config.attrs=&attr;config.numAttrs=1;
    std::vector<double> latency;
    auto stat=b.Time([&]{Check(cudaLaunchKernelEx(&config,ClusterHop,data.p,std::size_t(kib)*1024,cluster,background,b.Counters(),b.Times(),b.Sink()));},
                    [&]{latency.push_back(HopLatency(b,foreground));});
    b.Add("MB-1d-cluster","\"cluster_size\":"+std::to_string(cluster)+",\"KiB\":"+std::to_string(kib)+",\"background\":"+(background?"true":"false"),stat,double(kib)*1024*foreground,Median(latency),"hop_ns",latency);
  }
  else b.Skip("MB-1d-cluster","arch::Caps::kCluster is false");
}
void Steps(Bench& b) {
  int grid=b.device.multiProcessorCount,shared=ResidentShared(b);Buffer data(16*MiB);
  Check(cudaFuncSetAttribute(StepStream,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  auto baseline=b.Time([&]{StepStream<<<grid,128,shared>>>(data.p,data.bytes,b.Sink(),b.Counters(),1,false,0,false);});
  for(int mode=0;mode<3;++mode) {
    std::vector<cudaEvent_t> events;
    if(mode==1){events.resize(255);for(auto& event:events)Check(cudaEventCreate(&event));}
    auto stat=b.Time([&]{
      if(mode==2)StepStream<<<grid,128,shared>>>(data.p,data.bytes,b.Sink(),b.Counters(),256,false,0,false);
      else for(int step=0;step<256;++step){StepStream<<<grid,128,shared>>>(data.p,data.bytes,b.Sink(),b.Counters(),1,false,0,false);if(mode==1 && step<255)Check(cudaEventRecord(events[step]));}
    });
    double overhead=(stat.ms-256*baseline.ms)*1e6/256;
    b.Add("MB-1e","\"mode\":"+Quote(mode==0?"separate":mode==1?"events":"persistent")+",\"stream_ms\":"+std::to_string(baseline.ms),stat,data.bytes*256,overhead,"overhead_ns_per_step");
    for(auto event:events)Check(cudaEventDestroy(event));
  }
  if(b.caps.pdl)for(int trigger:{0,1})for(bool early:{false,true}) {
    auto stat=b.Time([&]{for(int i=0;i<256;++i)Check(ex::LaunchServing(StepStream,grid,128,shared,nullptr,true,data.p,data.bytes,b.Sink(),b.Counters(),1,true,trigger,early));});
    b.Add("MB-1e-PDL","\"trigger\":"+std::to_string(trigger)+",\"early_prefetch\":"+(early?"true":"false"),stat,data.bytes*256,(stat.ms-256*baseline.ms)*1e6/256,"overhead_ns_per_step");
  }else b.Skip("MB-1e-PDL","arch::Caps::kPdl is false");
}
void Cache(Bench& b) {
  Buffer weights(2048*MiB);cudaEvent_t ks,ke;Check(cudaEventCreate(&ks));Check(cudaEventCreate(&ke));
  for(int mib:{128,256}) {
    Buffer kv(mib*MiB);
    for(bool first:{false,true}) {
      std::vector<double> kv_times;
      auto total=b.Time([&]{
        CpLaunch(8,first?3:0,weights.p,weights.bytes/16,1,b.device.multiProcessorCount,256);
        Check(cudaEventRecord(ks));Stream(b,1,0,kv.p,kv.bytes,2,b.device.multiProcessorCount*4,256);Check(cudaEventRecord(ke));
      },[&]{float ms;Check(cudaEventElapsedTime(&ms,ks,ke));kv_times.push_back(ms);});
      b.Add("MB-1f","\"KV_MiB\":"+std::to_string(mib)+",\"evict_first\":"+(first?"true":"false")+",\"total_ms\":"+std::to_string(total.ms),Stat{Median(kv_times),kv_times},kv.bytes*2);
    }
  }Check(cudaEventDestroy(ks));Check(cudaEventDestroy(ke));
}
} // namespace
int main(int argc,char** argv) {
  std::string suite="all",output="loadbench.json";bool replay=false;Method best{3,8,0,256};
  try {
    for(int i=1;i<argc;++i) {
      std::string arg=argv[i];if(arg=="--suite")suite=argv[++i];else if(arg=="--out")output=argv[++i];
      else if(arg=="--ceiling-replay")replay=true;
      else if(arg=="--best-method")best.method=std::stoi(argv[++i]);else if(arg=="--best-param")best.param=std::stoi(argv[++i]);
      else if(arg=="--best-grid")best.grid=std::stoi(argv[++i]);else if(arg=="--best-threads")best.threads=std::stoi(argv[++i]);
      else if(arg=="--help") {std::cout<<"tilemega-loadbench --suite a|b|c|d|e|f|all --out FILE [--ceiling-replay --best-method M --best-param P --best-grid G --best-threads T]\n";return 0;}
      else throw std::invalid_argument("unknown option "+arg);
    }
    auto before=Observe();Bench b;if(!best.grid)best.grid=b.device.multiProcessorCount;
    if(suite=="all"||suite=="a")Ceiling(b,replay,best);
    if(suite=="all"||suite=="b")Pages(b);
    if(suite=="all"||suite=="c")Shapes(b);
    if(suite=="all"||suite=="d")Hops(b);
    if(suite=="all"||suite=="e")Steps(b);
    if(suite=="all"||suite=="f")Cache(b);
    auto after=Observe();std::ofstream out(output);
    out<<"{\"schema\":1,\"device\":"<<Quote(b.device.name)<<",\"arch\":"<<Quote("sm_"+std::to_string(b.device.major*10+b.device.minor))<<",\"num_sms\":"<<b.device.multiProcessorCount<<",\"smem_optin_bytes\":"<<b.device.sharedMemPerBlockOptin<<",\"repeats\":10,\"pid\":"<<getpid()<<",\"contaminated\":"<<(before.external||after.external?"true":"false")<<",\"before\":{\"clocks\":"<<Quote(before.clocks)<<",\"processes\":"<<Quote(before.apps)<<"},\"after\":{\"clocks\":"<<Quote(after.clocks)<<",\"processes\":"<<Quote(after.apps)<<"},\"points\":[";
    for(std::size_t i=0;i<b.points.size();++i){if(i)out<<',';out<<b.points[i];}out<<"]}\n";out.close();
    if(!out)throw std::runtime_error("cannot write output");
    std::cout<<"loadbench suite="<<suite<<" points="<<b.points.size()<<" contaminated="<<(before.external||after.external)<<'\n';return 0;
  }catch(std::exception const& e){std::cerr<<"loadbench error: "<<e.what()<<'\n';return 1;}
}
