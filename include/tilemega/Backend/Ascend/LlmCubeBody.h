// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Ascend/GemmTile.h>
#include <tilemega/Backend/Ascend/LlmAttentionShape.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>

namespace tilemega::ascend {
template<unsigned TM,unsigned TN,unsigned TK,unsigned Variant> class LlmCubeBody {
  using Bf16=GemmTile<bfloat16_t,bfloat16_t,TM,TN,TK,true>;
#if defined(TILEMEGA_LLM_SWIGLU_L1_K256) && TILEMEGA_LLM_SWIGLU_L1_K256
  using Bf16K256=GemmTile<bfloat16_t,bfloat16_t,16,128,256,true,32>;
#endif
#if (defined(TILEMEGA_LLM_SPLIT_K) && TILEMEGA_LLM_SPLIT_K) || (defined(TILEMEGA_LLM_LOCAL_TAIL_K7) && TILEMEGA_LLM_LOCAL_TAIL_K7) || (defined(TILEMEGA_LLM_LOCAL_Q_TAIL_K3) && TILEMEGA_LLM_LOCAL_Q_TAIL_K3)
  using Partial=GemmTile<bfloat16_t,float,TM,TN,TK,true,64>;
#endif
  using Qk=GemmTile<float,float,32,64,128,true>;
#if defined(TILEMEGA_LLM_WIDE_QK) && TILEMEGA_LLM_WIDE_QK
  using QkWide=GemmTile<float,float,32,256,128,true>;
#endif
  using Pv=GemmTile<float,float,32,128,64,false>;
  typename Bf16::SharedResource resource_;
 public:
  __aicore__ inline bool Run(llm::Stage const& s,unsigned tile,GM_ADDR arena,unsigned owner_group) {
    if(s.kind!=0 || tile>=s.tasks || (s.variant!=Variant && !((s.variant>=2 && s.variant<=6) && Variant==1)) || s.k%16)return false;
    if(s.variant==6) {
#if defined(TILEMEGA_LLM_VECTOR_ARGMAX) && TILEMEGA_LLM_VECTOR_ARGMAX
      if(TM!=16 || TN!=128 || s.epilogue!=3 || s.extent>1 || !s.rows || s.rows>16)return false;
#else
      return false;
#endif
    }
    if(s.variant==4 || s.variant==5) {
#if defined(TILEMEGA_LLM_SWIGLU_TILE) && TILEMEGA_LLM_SWIGLU_TILE
      const bool family=(s.columns==6144 && s.k==1024) || (s.columns==12288 && s.k==2048) ||
          (s.columns==16384 && (s.k==2048 || s.k==3072)) || (s.columns==28672 && s.k==4096);
      const bool rows=s.rows==1 || s.rows==16 || s.rows==64 || s.rows==1024;
      if(TM!=16 || TN!=128 || !rows || (s.variant==5 && s.rows!=16) || !family || s.epilogue!=2 || s.extent>1)return false;
#else
      return false;
#endif
    }
    unsigned nt=(s.columns+TN-1)/TN;
    if(s.variant==5) {
#if defined(TILEMEGA_LLM_SWIGLU_L1_K256) && TILEMEGA_LLM_SWIGLU_L1_K256
      // Only internal L1 staging changes; the CTG tile and L0 reduction stay fixed.
      typename Bf16K256::TileMmad mm(resource_);
      Bf16K256::RunBorrowed(mm,arena+s.operands[0],arena+s.operands[1],arena+s.workspace,
          s.rows,s.columns,s.k,tile/nt,tile%nt);
      CompleteDma<AscendC::HardEvent::FIX_S>();return true;
#else
      return false;
#endif
    }
#if defined(TILEMEGA_LLM_LOCAL_TAIL_K7) && TILEMEGA_LLM_LOCAL_TAIL_K7
    if(s.variant==2) {
      if(TM!=16 || TN!=128 || s.rows!=1024 || s.columns!=2048 || (s.k!=8192 && s.k!=6144) ||
          s.extent>1 || s.epilogue!=1 || s.operands[5]==llm::kAbsent)return false;
      if(tile%nt*TN>=1792) {
        typename Partial::TileMmad mm(resource_);
        auto partial=arena+s.operands[5]+uint64_t(owner_group)*7*TM*TN*4;
        for(unsigned part=0;part<7;++part) {
          unsigned width=(s.k+6)/7,begin=part*width,size=s.k-begin<width?s.k-begin:width;
          Partial::template RunKSliceBorrowed<true>(mm,arena+s.operands[0],arena+s.operands[1],
            partial+uint64_t(part)*TM*TN*4,s.rows,s.columns,s.k,begin,size,tile/nt,tile%nt);
          CompleteDma<AscendC::HardEvent::FIX_S>();
        }
        return true;
      }
    }
#else
    if(s.variant==2)return false;
#endif
#if defined(TILEMEGA_LLM_LOCAL_Q_TAIL_K3) && TILEMEGA_LLM_LOCAL_Q_TAIL_K3
    if(s.variant==3) {
      const bool store=s.k==4096 && s.epilogue==4 && s.workspace==s.operands[3];
      const bool residual=(s.k==4096 || s.k==14336) && s.epilogue==1 && s.workspace!=llm::kAbsent;
      if(TM!=16 || TN!=128 || s.rows!=1024 || s.columns!=4096 ||
          s.extent>1 || (!store && !residual) || s.operands[5]==llm::kAbsent)return false;
      if(tile%nt*TN>=3584) {
        typename Partial::TileMmad mm(resource_);
        auto partial=arena+s.operands[5]+uint64_t(owner_group)*3*TM*TN*4;
        for(unsigned part=0;part<3;++part) {
          const unsigned width=(s.k+2)/3,begin=part*width,size=s.k-begin<width?s.k-begin:width;
          Partial::template RunKSliceBorrowed<true>(mm,arena+s.operands[0],arena+s.operands[1],
              partial+uint64_t(part)*TM*TN*4,s.rows,s.columns,s.k,begin,size,tile/nt,tile%nt);
          CompleteDma<AscendC::HardEvent::FIX_S>();
        }
        return true;
      }
    }
#else
    if(s.variant==3)return false;
#endif
#if defined(TILEMEGA_LLM_SPLIT_K) && TILEMEGA_LLM_SPLIT_K
    if(s.extent>1) {
      const bool family=(s.extent==3 && s.columns==2048 && (s.k==6144 || s.k==8192)) ||
                        (s.extent==2 && s.columns==3072 && s.k==8192) ||
                        (s.extent==7 && s.columns==1024 && s.k==4096);
      if(!family || s.width!=(s.k+s.extent-1)/s.extent ||
          (s.rows!=1 && s.rows!=64 && !(s.rows==16 && ((s.extent==3 && s.columns==2048 && (s.k==8192 || s.k==6144)) || (s.extent==2 && s.columns==3072 && s.k==8192) || (s.extent==7 && s.columns==1024 && s.k==4096)))) || s.epilogue || s.workspace==llm::kAbsent || !s.group)return false;
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
    auto key=s.variant==9 && m.seq==1?LlmAttentionCachedKv(arena,s,m,h,false):work+llm::attention::kK;
    auto value=s.variant==9 && m.seq==1?LlmAttentionCachedKv(arena,s,m,h,true):work+llm::attention::kV;
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    if(!pv) {
      if(s.variant==7) {
#if defined(TILEMEGA_LLM_WIDE_QK) && TILEMEGA_LLM_WIDE_QK
        typename QkWide::TileMmad mm(resource_);
        for(unsigned row=0;row<(h.rows+31)/32;++row)
          for(unsigned col=0;col<(h.keys+255)/256;++col) {
            QkWide::RunBorrowed(mm,work+llm::attention::kQ,work+llm::attention::kK,
              work+llm::attention::kScores,h.rows,h.keys,h.width,row,col);
            CompleteDma<AscendC::HardEvent::FIX_S>();
          }
        return true;
#else
        return false;
#endif
      }
      typename Qk::TileMmad mm(resource_);
      // Softmax DMA reads valid_keys only and constructs its masked tail locally.
      // Keep the original physical stride/tile, including its last partial tile;
      // only entirely unconsumed score columns are omitted.
      const unsigned qk_keys=s.variant==8 || s.variant==9?h.valid_keys:h.keys;
      for(unsigned row=0;row<(h.rows+31)/32;++row)
        for(unsigned col=0;col<(qk_keys+63)/64;++col) {
          Qk::RunBorrowed(mm,work+llm::attention::kQ,key,
            work+llm::attention::kScores,h.rows,h.keys,h.width,row,col);
          CompleteDma<AscendC::HardEvent::FIX_S>();
        }
    }else {
      typename Pv::TileMmad mm(resource_);
      for(unsigned row=0;row<(h.rows+31)/32;++row) {
        Pv::RunBorrowed(mm,work+llm::attention::kProbabilities,value,
          work+llm::attention::kOutput,h.rows,h.width,h.keys,row,0);
        CompleteDma<AscendC::HardEvent::FIX_S>();
      }
    }
    return true;
  }
};
} // namespace tilemega::ascend
