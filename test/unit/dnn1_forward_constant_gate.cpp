// Synthetic CPU lookup contracts; these values are not calibration data.
#include <tilemega/Solver/Sm90ForwardSearch.h>
#include <cassert>
#include <fstream>
using namespace tilemega;
using namespace tilemega::solver;
int main(int argc,char** argv) {
  assert(argc==2);setenv("DNN1_CPU_CONTRACT_TEST","1",1);
  std::string key="tw=8;th=8;channels=32;ct=64;stride=1;pack=8";
  std::string gated=key+";gate=simple_gate;bias=1";
  auto selector=Sm90ForwardConstants::ParseKey(key);
  for(int schema:{1,2}) {
    json::Value section=json::Object{{"schema",schema},{"variant","SYNTHETIC_CPU_ONLY"},
        {"synthetic_cpu_fixture",true}};
    if(schema==1) {
      json::Object table;
      for(auto const& entry:{std::pair{key,2.0},std::pair{gated,9.0}})
        table.emplace_back(entry.first,json::Object{{"median",entry.second},
            {"processes",5},{"unit","ns/element/tap"},{"unstable",false}});
      section.Set("by_gpu",json::Object{{"0",json::Object{{"dw_tap_ns",table}}}});
    }else {
      section.Set("keys",json::Array{json::Array{"dw_tap_ns",key,"ns/element/tap"},
          json::Array{"dw_tap_ns",gated,"ns/element/tap"}});
      section.Set("median_by_gpu",json::Object{{"0",json::Array{2,9}},
          {"5",json::Array{2,9}},{"7",json::Array{2,9}}});
      section.Set("processes",5);section.Set("unstable",json::Array{});
    }
    json::Value target=json::Object{{"sm90_forward_constants",section}};
    {std::ofstream out(argv[1]);out<<target.Dump(0);}
    auto constants=Sm90ForwardConstants::FromFile(argv[1],0);
    assert(constants.Get("dw_tap_ns",selector,"ns/element/tap").ns==2);
    auto explicit_gate=selector;explicit_gate["gate"]="simple_gate";
    assert(constants.Get("dw_tap_ns",explicit_gate,"ns/element/tap").ns==9);
    explicit_gate["gate"]="missing";bool rejected=false;
    try{constants.Get("dw_tap_ns",explicit_gate,"ns/element/tap");}
    catch(Sm90ForwardMissing const&){rejected=true;}
    assert(rejected);
  }
}
