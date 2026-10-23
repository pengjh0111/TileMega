// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <stdexcept>
#include <string_view>

namespace tilemega::codegen::native_dense::boundary_diagnostic {

enum class Class : unsigned {Embedding,Qkv,Attention,O,GateUp,Down,Head,Argmax};
inline constexpr unsigned kClasses=8,kAll=(1u<<kClasses)-1;
inline constexpr char const* kNames[kClasses]={
    "embedding","qkv","attention","o","gate_up","down","lm_head","argmax"};
inline constexpr bool ValidStages(unsigned count) {return count>=8 && (count-3)%5==0;}
inline constexpr Class ClassOf(unsigned index,unsigned count) {
  return index==0?Class::Embedding:index==count-1?Class::Argmax:
      index==count-2?Class::Head:static_cast<Class>(1+(index-1)%5);
}
inline constexpr unsigned Bit(Class stage) {return 1u<<static_cast<unsigned>(stage);}

struct Config {
  unsigned observe=0,probe_mask=0;
  char const* mode="off";
  char const* selected_class="";
};

inline Config Parse(std::string_view mode,std::string_view selected={}) {
  if(mode=="only-class" || mode=="leave-class-out") {
    unsigned bit=0;
    for(unsigned i=0;i<kClasses;++i)if(selected==kNames[i])bit=1u<<i;
    if(!bit)throw std::invalid_argument("boundary diagnostic requires a known class");
    return {1,mode=="only-class"?bit:kAll^bit,
        mode=="only-class"?"only-class":"leave-class-out",
        kNames[static_cast<unsigned>(__builtin_ctz(bit))]};
  }
  if(!selected.empty())throw std::invalid_argument("boundary diagnostic class requires a class mode");
  if(mode=="off")return {};
  if(mode=="lean")return {1,0,"lean",""};
  if(mode=="full")return {1,kAll,"full",""};
  throw std::invalid_argument("unknown native boundary diagnostic mode");
}

inline bool Matches(Class stage,std::string_view kind,std::string_view name) {
  switch(stage) {
    case Class::Embedding:return kind=="kEmbedding";
    case Class::Attention:return kind=="kFusedAttention";
    case Class::Argmax:return kind=="kArgmaxReduce";
    default:break;
  }
  if(kind!="kGemm")return false;
  auto token=stage==Class::Qkv?"qkv":stage==Class::O?"o_proj":
      stage==Class::GateUp?"gate_up":stage==Class::Down?"down_proj":"lm_head";
  return name.find(token)!=std::string_view::npos;
}

}  // namespace tilemega::codegen::native_dense::boundary_diagnostic
