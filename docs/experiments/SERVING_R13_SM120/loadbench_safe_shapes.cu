// SPDX-License-Identifier: BSD-3-Clause
// Reuse the registered kernels unchanged; quarantine one observed device fault.
#define main ArchivedLoadbenchMain
#include "../../../tools/experimental/loadbench/main.cu"
#undef main

int main(int argc, char** argv) {
  try {
    if (argc != 3 || std::string(argv[1]) != "--out")
      throw std::invalid_argument("safe-shapes --out FILE");
    auto before = Observe();
    Bench b;
    if (b.device.major != 12 || b.device.minor != 0)
      throw std::invalid_argument("this fault quarantine is only registered for sm120");
    Buffer data(2048 * MiB);
    for (auto geometry : std::vector<std::tuple<int,int,int>>{{32,128,4},{32,128,7},{128,64,4}}) {
      int tn=std::get<0>(geometry), tk=std::get<1>(geometry), s=std::get<2>(geometry);
      if (tn*tk*2*s > b.device.sharedMemPerBlockOptin) continue;
      for (int k : {2048,8192}) for (int active : {50,75,100})
        for (bool row : {false,true}) for (int method=0; method<6; ++method) {
          if (tn == 128 && tk == 64 && method == 5) {
            b.Skip("MB-1c", "TN128/TK64 method5 quarantined after an illegal instruction and ownerless GPU saturation; not a pass");
            continue;
          }
          int grid=std::max(1,b.device.multiProcessorCount*active/100);
          auto stat=b.Time([&]{
            if(tn==128) Shape<128,64,4>(b,data.p,data.bytes,k,grid,row,method);
            else if(s==4) Shape<32,128,4>(b,data.p,data.bytes,k,grid,row,method);
            else Shape<32,128,7>(b,data.p,data.bytes,k,grid,row,method);
          });
          b.Add("MB-1c","\"tile_n\":"+std::to_string(tn)+",\"tile_k\":"+std::to_string(tk)+
                ",\"stages\":"+std::to_string(s)+",\"K\":"+std::to_string(k)+
                ",\"active_pct\":"+std::to_string(active)+",\"row\":"+(row?"true":"false")+
                ",\"method\":"+std::to_string(method),stat,data.bytes);
        }
    }
    auto after=Observe();
    std::ofstream out(argv[2]);
    out << "{\"schema\":1,\"device\":" << Quote(b.device.name)
        << ",\"arch\":\"sm_120\",\"num_sms\":" << b.device.multiProcessorCount
        << ",\"smem_optin_bytes\":" << b.device.sharedMemPerBlockOptin
        << ",\"repeats\":10,\"pid\":" << getpid()
        << ",\"contaminated\":" << (before.external||after.external?"true":"false")
        << ",\"quarantine\":\"TN128/TK64 method5\",\"before\":{\"clocks\":" << Quote(before.clocks)
        << ",\"processes\":" << Quote(before.apps) << "},\"after\":{\"clocks\":" << Quote(after.clocks)
        << ",\"processes\":" << Quote(after.apps) << "},\"points\":[";
    for (std::size_t i=0; i<b.points.size(); ++i) { if(i)out<<','; out<<b.points[i]; }
    out << "]}\n";
    out.close();
    if (!out) throw std::runtime_error("cannot write output");
    std::cout << "MB-1c completed supported points; 12 quarantined points are not passes\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "safe-shapes error: " << error.what() << '\n';
    return 1;
  }
}
