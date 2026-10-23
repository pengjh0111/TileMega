// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Ascend/GemmTile.h>
#include <tilemega/Backend/Ascend/LlmAttentionShape.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>

namespace tilemega::ascend {
template<unsigned TM,unsigned TN,unsigned TK,unsigned Variant> class LlmCubeBody {
  using Bf16=GemmTile<bfloat16_t,bfloat16_t,TM,TN,TK,true>;
#if defined(TILEMEGA_LLM_SPLIT_K) && TILEMEGA_LLM_SPLIT_K
  using Partial=GemmTile<bfloat16_t,float,TM,TN,TK,true,64>;
#endif
  using Qk=GemmTile<float,float,32,64,128,true>;
  using Pv=GemmTile<float,float,32,128,64,false>;
  typename Bf16::SharedResource resource_;
 public:
  __aicore__ inline bool Run(llm::Stage const& s,unsigned tile,GM_ADDR arena) {
    if(s.kind!=0 || tile>=s.tasks || s.variant!=Variant || s.k%16)return false;
    unsigned nt=(s.columns+TN-1)/TN;
#if defined(TILEMEGA_LLM_SPLIT_K) && TILEMEGA_LLM_SPLIT_K
    if(s.extent>1) {
      const bool family=(s.extent==3 && s.columns==2048 && (s.k==6144 || s.k==8192)) ||
                        (s.extent==2 && s.columns==3072 && s.k==8192) ||
                        (s.extent==7 && s.columns==1024 && s.k==4096);
      if(!family || s.width!=(s.k+s.extent-1)/s.extent ||
          (s.rows!=1 && s.rows!=64) || s.epilogue || s.workspace==llm::kAbsent || !s.group)return false;
      const unsigned part=tile%s.extent,output_tile=tile/s.extent;
      typename Partial::TileMmad mm(resource_);
      const unsigned begin=part*s.width,size=s.k-begin<s.width?s.k-begin:s.width;
      Partial::RunKSliceBorrowed(mm,arena+s.operands[0],arena+s.operands[1],
        arena+s.workspace+uint64_t(part)*s.rows*s.columns*sizeof(float),s.rows,s.columns,s.k,
        begin,size,output_tile/nt,output_tile%nt);
      CompleteDma<AscendC::HardEvent::FIX_S>();return true;
    }
#endif
    typename Bf16::TileMmad mm(resource_);
    Bf16::RunBorrowed(mm,arena+s.operands[0],arena+s.operands[1],arena+(s.epilogue?s.workspace:s.operands[3]),
      s.rows,s.columns,s.k,tile/nt,tile%nt);
    CompleteDma<AscendC::HardEvent::FIX_S>();return true;
  }
  __aicore__ inline bool AttentionGemm(llm::Stage const& s,unsigned tile,GM_ADDR arena,
      llm::Metadata const& m,unsigned past,unsigned owner_group,bool pv) {
    LlmAttentionShape h;
    if(owner_group>=m.groups || !DescribeLlmAttention(s,tile,m,past,h))return false;
    auto work=LlmAttentionWork(arena,s,owner_group);
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    if(!pv) {
      typename Qk::TileMmad mm(resource_);
      for(unsigned row=0;row<(h.rows+31)/32;++row)
        for(unsigned col=0;col<(h.keys+63)/64;++col) {
          Qk::RunBorrowed(mm,work+llm::attention::kQ,work+llm::attention::kK,
            work+llm::attention::kScores,h.rows,h.keys,h.width,row,col);
          CompleteDma<AscendC::HardEvent::FIX_S>();
        }
    }else {
      typename Pv::TileMmad mm(resource_);
      for(unsigned row=0;row<(h.rows+31)/32;++row) {
        Pv::RunBorrowed(mm,work+llm::attention::kProbabilities,work+llm::attention::kV,
          work+llm::attention::kOutput,h.rows,h.width,h.keys,row,0);
        CompleteDma<AscendC::HardEvent::FIX_S>();
      }
    }
    return true;
  }
};
} // namespace tilemega::ascend
