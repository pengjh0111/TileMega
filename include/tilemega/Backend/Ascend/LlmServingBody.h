// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Ascend/LlmVectorBody.h>
#include <tilemega/Target/Ascend/LlmPlanABI.h>
#include <tilemega/Backend/Ascend/LlmAttentionShape.h>

namespace tilemega::ascend {
template<unsigned TM=32,unsigned TN=64,bool CaptureSoftmax=false> class LlmServingBody {
  struct Operands {
    GM_ADDR arena;llm::Stage const& stage;
    __aicore__ inline GM_ADDR operator()(unsigned i) const {
      return stage.operands[i]==llm::kAbsent?nullptr:arena+stage.operands[i];
    }
  };
  LlmVectorBody vector_;
  __aicore__ inline void Acquire(GM_ADDR p) {
    AscendC::GlobalTensor<uint8_t> tensor;tensor.SetGlobalBuffer(p);AcquireData(tensor);
  }
  __aicore__ inline bool Rotate(GM_ADDR source,GM_ADDR gamma,GM_ADDR cos,GM_ADDR sin,
      GM_ADDR output,GM_ADDR temporary,unsigned d,float epsilon,unsigned norm_variant=0) {
    if(gamma) {
      if(!vector_.RmsNorm(source,gamma,temporary,d,epsilon,norm_variant))return false;
      Acquire(temporary);source=temporary;
    }
    return vector_.Rope(source,cos,sin,output,d);
  }
  __aicore__ inline bool Attend(GM_ADDR query,GM_ADDR key,GM_ADDR value,GM_ADDR output,
      GM_ADDR scratch,unsigned keys,unsigned d,unsigned precision) {
    if(precision==1) {
      unsigned chunks=0;
      for(unsigned k=0;k<keys;k+=32,++chunks) {
        unsigned n=keys-k<32?keys-k:32;
        if(!vector_.AttentionChunk(query,key+uint64_t(k)*d*2,value+uint64_t(k)*d*2,
            scratch+1024+chunks*d*4,scratch+20480+chunks*8,n,d,
            d==128?0.08838834764831845f:0.125f))return false;
      }
      Acquire(scratch);
      return vector_.AttentionMerge(scratch+1024,scratch+20480,output,chunks,d);
    }
    for(unsigned k=0;k<keys;k+=32) {
      unsigned n=keys-k<32?keys-k:32;
      if(!vector_.ServingScores(query,key+uint64_t(k)*d*2,scratch+1024+k*4,n,d,
          d==128?0.08838834764831845f:0.125f,precision==0,precision==3))return false;
    }
    Acquire(scratch);
    if(!vector_.ServingSoftmax(scratch+1024,scratch+6144,keys,precision==3,precision==3?1088:keys))return false;
    Acquire(scratch);unsigned count=0;
    unsigned partial=precision==3?16384:10240;
    for(unsigned k=0;k<keys;k+=32,++count) {
      unsigned n=keys-k<32?keys-k:32;
      if(!vector_.ServingPV(scratch+6144+k*(precision==3?4:2),value+uint64_t(k)*d*2,scratch+partial+count*d*4,n,d,precision==3))return false;
    }
    Acquire(scratch);return vector_.SumServingPV(scratch+partial,output,count,d);
  }
 public:
  __aicore__ inline bool AttentionPrepare(llm::Stage const& s,unsigned tile,GM_ADDR arena,
      GM_ADDR scratch,llm::Metadata const& m,unsigned past,unsigned owner_group) {
    LlmAttentionShape h;
    if(owner_group>=m.groups || !DescribeLlmAttention(s,tile,m,past,h))return false;
    Operands p{arena,s};auto work=LlmAttentionWork(arena,s,owner_group);
    unsigned d=h.width;uint64_t packed=uint64_t(s.extent)*(s.group+2)*d;
    GM_ADDR keys=nullptr;
    GM_ADDR values=nullptr;
    if(m.seq==64) {
      keys=scratch+32768;values=scratch+65536;
      for(unsigned pos=0;pos<64;++pos) {
        auto source=p(0)+((uint64_t(h.batch)*64+pos)*packed+uint64_t(h.kv_group)*(s.group+2)*d)*2;
        if(!Rotate(source+s.group*d*2,p(6),p(3)+uint64_t(pos)*d*2,p(4)+uint64_t(pos)*d*2,
            keys+uint64_t(pos)*d*2,scratch+512,d,s.epsilon,s.variant>=5?2:0) ||
            !vector_.Copy(source+(s.group+1)*d*2,values+uint64_t(pos)*d*2,d))return false;
        if(pos>=h.first_position && pos<h.first_position+h.positions) {
          auto offset=((uint64_t(h.batch)*s.extent+h.kv_group)*m.capacity+pos)*d*2;
          if(!vector_.Copy(keys+uint64_t(pos)*d*2,p(1)+offset,d) ||
              !vector_.Copy(values+uint64_t(pos)*d*2,p(2)+offset,d))return false;
        }
      }
    }else {
      auto source=p(0)+(uint64_t(h.batch)*packed+uint64_t(h.kv_group)*(s.group+2)*d)*2;
      keys=p(1)+(uint64_t(h.batch)*s.extent+h.kv_group)*m.capacity*d*2;
      values=p(2)+(uint64_t(h.batch)*s.extent+h.kv_group)*m.capacity*d*2;
      if(!Rotate(source+s.group*d*2,p(6),p(3)+uint64_t(past)*d*2,p(4)+uint64_t(past)*d*2,
          keys+uint64_t(past)*d*2,scratch+512,d,s.epsilon,s.variant>=5?2:0) ||
          !vector_.Copy(source+(s.group+1)*d*2,values+uint64_t(past)*d*2,d))return false;
    }
    Acquire(scratch);
    const float root=d==128?0.29730177875068026f:0.3535533905932738f;
    unsigned valid=h.valid_keys*d,total=h.keys*d;
    for(unsigned first=0;first<valid;first+=4096) {
      unsigned n=valid-first<4096?valid-first:4096;
      if(!vector_.Bf16ToFloat(keys+uint64_t(first)*2,work+llm::attention::kK+uint64_t(first)*4,n,root) ||
          !vector_.Bf16ToFloat(values+uint64_t(first)*2,work+llm::attention::kV+uint64_t(first)*4,n))return false;
    }
    // Decode never reads invalid public KV. Its physical FP32 matmul tail is
    // explicitly zeroed before use, including workspace recycled from prefill.
    for(unsigned first=valid;first<total;first+=4096) {
      unsigned n=total-first<4096?total-first:4096;
      if(!vector_.ZeroFloat(work+llm::attention::kK+uint64_t(first)*4,n) ||
          !vector_.ZeroFloat(work+llm::attention::kV+uint64_t(first)*4,n))return false;
    }
    for(unsigned row=0;row<h.rows;++row) {
      unsigned pos=h.first_position+row/s.group,q=row%s.group;
      auto source=p(0)+((uint64_t(h.batch)*m.seq+(m.seq==64?pos:0))*packed+
        uint64_t(h.kv_group)*(s.group+2)*d+q*d)*2;
      if(!Rotate(source,p(5),p(3)+uint64_t(pos)*d*2,p(4)+uint64_t(pos)*d*2,scratch,scratch+512,d,s.epsilon,s.variant>=5?2:0))return false;
      Acquire(scratch);
      if(!vector_.Bf16ToFloat(scratch,work+llm::attention::kQ+uint64_t(row)*d*4,d,root))return false;
    }
    return true;
  }
  __aicore__ inline bool AttentionSoftmax(llm::Stage const& s,unsigned tile,GM_ADDR arena,
      llm::Metadata const& m,unsigned past,unsigned owner_group) {
    LlmAttentionShape h;
    if(owner_group>=m.groups || !DescribeLlmAttention(s,tile,m,past,h))return false;
    auto work=LlmAttentionWork(arena,s,owner_group);
    for(unsigned row=0;row<h.rows;++row) {
      unsigned valid=m.seq==64?h.first_position+row/s.group+1:past+1;
      GM_ADDR diagnostic_exp=nullptr;
      GM_ADDR diagnostic_stats=nullptr;
      if constexpr(CaptureSoftmax) {
        diagnostic_exp=work+llm::attention::kDiagnosticExp+uint64_t(row)*m.capacity*4;
        diagnostic_stats=work+llm::attention::kDiagnosticStats+uint64_t(row)*64;
      }
      if(!vector_.ServingSoftmax(work+llm::attention::kScores+uint64_t(row)*h.keys*4,
          work+llm::attention::kProbabilities+uint64_t(row)*h.keys*4,valid,true,m.capacity,h.keys,true,
          diagnostic_exp,diagnostic_stats,s.variant==6 || s.variant==7 || s.variant==8))return false;
    }
    return true;
  }
  __aicore__ inline bool AttentionFinish(llm::Stage const& s,unsigned tile,GM_ADDR arena,
      llm::Metadata const& m,unsigned past,unsigned owner_group) {
    LlmAttentionShape h;
    if(owner_group>=m.groups || !DescribeLlmAttention(s,tile,m,past,h))return false;
    auto work=LlmAttentionWork(arena,s,owner_group);Operands p{arena,s};
    unsigned width=s.group*h.width;
    for(unsigned i=0;i<h.positions;++i) {
      uint64_t position=uint64_t(h.batch)*m.seq+(m.seq==64?h.first_position+i:0);
      auto output=p(7)+(position*s.extent+h.kv_group)*width*2;
      if(!vector_.FloatToBf16(work+llm::attention::kOutput+uint64_t(i)*width*4,output,width))return false;
    }
    return true;
  }
  __aicore__ inline bool Epilogue(llm::Stage const& s,unsigned tile,GM_ADDR arena,unsigned owner_group=0) {
    unsigned nt=(s.columns+TN-1)/TN,first=(tile%nt)*TN,rows=tile/nt*TM;
    if(s.variant==3) {
#if defined(TILEMEGA_LLM_LOCAL_Q_TAIL_K3) && TILEMEGA_LLM_LOCAL_Q_TAIL_K3
      const bool store=s.k==4096 && s.epilogue==4 && s.workspace==s.operands[3];
      const bool residual=(s.k==4096 || s.k==14336) && s.epilogue==1 && s.workspace!=llm::kAbsent;
      if(TM!=16 || TN!=128 || s.rows!=1024 || s.columns!=4096 ||
          s.extent>1 || (!store && !residual) || s.operands[5]==llm::kAbsent)return false;
      // Regular tiles already wrote BF16; residual rounding follows the ordered sum.
      if(first>=3584)for(unsigned r=rows;r<rows+TM && r<s.rows;++r) {
        auto partial=arena+s.operands[5]+(uint64_t(owner_group)*3*TM*TN+(r-rows)*TN)*4;
        auto output=arena+s.workspace+(uint64_t(r)*s.columns+first)*2;
        if(!vector_.CombineGemm(partial,output,TM*TN,3,TN))return false;
        Acquire(output);
      }
      if(store)return true;
#else
      return false;
#endif
    }
    for(unsigned r=rows;r<rows+TM && r<s.rows;++r) {
      auto raw=arena+s.workspace+(uint64_t(r)*s.columns+first)*2;
      unsigned n=s.columns-first<TN?s.columns-first:TN;
#if defined(TILEMEGA_LLM_LOCAL_TAIL_K7) && TILEMEGA_LLM_LOCAL_TAIL_K7
      if(s.variant==2 && first>=1792) {
        if(TM!=16 || TN!=128 || s.rows!=1024 || s.columns!=2048 || (s.k!=8192 && s.k!=6144) || s.epilogue!=1 || s.operands[5]==llm::kAbsent)return false;
        auto partial=arena+s.operands[5]+(uint64_t(owner_group)*7*TM*TN+(r-rows)*TN)*4;
        if(!vector_.CombineGemm(partial,raw,TM*TN,7,n))return false;
        Acquire(raw);
      }
#endif
      if(s.epilogue==1) {
        auto offset=(uint64_t(r)*s.columns+first)*2;
        if(!vector_.Pointwise(raw,arena+s.operands[2]+offset,arena+s.operands[3]+offset,n,false))return false;
      }else if(s.epilogue==2) {
        if(first%32 || n%32)return false;
        for(unsigned col=0;col<n;col+=32) {
          auto output=arena+s.operands[3]+(uint64_t(r)*(s.columns/2)+(first+col)/2)*2;
          if(!vector_.Pointwise(raw+col*2,raw+(col+16)*2,output,16,true))return false;
        }
      }else if(s.epilogue==3) {
        auto slot=(uint64_t(r)*nt+tile%nt)*4;
        if(!vector_.ArgmaxTile(raw,arena+s.operands[3]+slot,arena+s.operands[4]+slot,n,first))return false;
      }else return false;
    }
    return true;
  }
  __aicore__ inline bool Run(llm::Stage const& s,unsigned task,GM_ADDR arena,
      GM_ADDR scratch,llm::Metadata const& meta,unsigned past) {
    Operands p{arena,s};
    if(task>=s.tasks)return false;
#if defined(TILEMEGA_LLM_SPLIT_K) && TILEMEGA_LLM_SPLIT_K
    if(s.kind==6) {
      const bool family=(s.extent==3 && s.columns==2048 && (s.k==6144 || s.k==8192)) ||
                        (s.extent==2 && s.columns==3072 && s.k==8192) ||
                        (s.extent==7 && s.columns==1024 && s.k==4096);
      if(!family || s.width!=(s.k+s.extent-1)/s.extent ||
          (s.rows!=1 && s.rows!=64 && !(s.rows==16 && ((s.extent==3 && s.columns==2048 && (s.k==8192 || s.k==6144)) || (s.extent==2 && s.columns==3072 && s.k==8192) || (s.extent==7 && s.columns==1024 && s.k==4096)))) || s.epilogue!=(s.extent==7?0u:1u) || s.operands[5]==llm::kAbsent)return false;
      const unsigned nt=(s.columns+TN-1)/TN,first=task%nt*TN,row=task/nt*TM;
      for(unsigned r=row;r<row+TM && r<s.rows;++r) {
        unsigned n=s.columns-first<TN?s.columns-first:TN;
        uint64_t index=uint64_t(r)*s.columns+first;
        if(!vector_.CombineGemm(arena+s.workspace+index*4,
            p(5)+index*2,uint64_t(s.rows)*s.columns,s.extent,n))return false;
      }
      if(!s.epilogue)return true;
      Acquire(p(5));auto epilogue=s;epilogue.workspace=s.operands[5];
      return Epilogue(epilogue,task,arena);
    }
#endif
    if(s.kind==25) {
      if(s.extent!=8 || s.width!=768 || s.group!=1 || s.query_rows!=4 || s.kv_block!=128 || task>=s.rows)return false;
      const uint64_t qrow=uint64_t(task)*4096,kvrow=uint64_t(task)*1024,outrow=uint64_t(task)*6144;
      for(unsigned g=0;g<8;++g) {
        if(!vector_.Copy(p(0)+(qrow+g*512)*2,p(3)+(outrow+g*768)*2,512) ||
            !vector_.Copy(p(1)+(kvrow+g*128)*2,p(3)+(outrow+g*768+512)*2,128) ||
            !vector_.Copy(p(2)+(kvrow+g*128)*2,p(3)+(outrow+g*768+640)*2,128))return false;
      }
      return true;
    }
    if(s.kind==11) {
      unsigned row=task,b=row/meta.seq,pos=past+row%meta.seq;
      unsigned id=ReadControl((__gm__ uint32_t*)p(0)+uint64_t(b)*meta.capacity+pos);
      if(id>=s.extent)return false;
      return vector_.Copy(p(1)+uint64_t(id)*s.width*2,p(2)+uint64_t(row)*s.width*2,s.width);
    }
    if(s.kind==1) {
      uint64_t row=s.batch_rows?uint64_t(task)*s.row_stride+s.row_offset:task;
      return vector_.RmsNorm(p(0)+row*s.width*2,p(1),p(2)+uint64_t(task)*s.width*2,s.width,s.epsilon,s.variant);
    }
    if(s.kind==13) {
      if((meta.seq!=1 && meta.seq!=64) || (s.width!=64 && s.width!=128) || !s.group || s.group>8 || s.variant>3 || (s.kv_block!=128 && s.kv_block!=1088))return false;
      if(meta.seq==64) {
        if(past || s.kv_block!=meta.capacity || meta.worker_scratch_bytes<98304 || !s.query_rows || s.query_rows%s.group)return false;
        unsigned qblocks=(meta.seq*s.group+s.query_rows-1)/s.query_rows;
        // CTG flattens [token_tile, kv_group], with the group varying fastest.
        unsigned qb=(task/s.extent)%qblocks,group=task%s.extent,b=task/(qblocks*s.extent),d=s.width;
        if(b>=meta.batch)return false;
        uint64_t packed=uint64_t(s.extent)*(s.group+2)*d;
        auto keys=scratch+32768,values=scratch+65536;
        // All causal K/V are transformed from QKV, whose dependency is a
        // conservative full-row read. No query task reads another task's
        // not-yet-appended cache row. Only the block containing qhead0 appends.
        for(unsigned pos=0;pos<meta.seq;++pos) {
          auto source=p(0)+((uint64_t(b)*meta.seq+pos)*packed+uint64_t(group)*(s.group+2)*d)*2;
          if(!Rotate(source+s.group*d*2,p(6),p(3)+uint64_t(pos)*d*2,p(4)+uint64_t(pos)*d*2,
              keys+uint64_t(pos)*d*2,scratch+512,d,s.epsilon) ||
              !vector_.Copy(source+(s.group+1)*d*2,values+uint64_t(pos)*d*2,d))return false;
          if(pos*s.group/s.query_rows==qb) {
            Acquire(scratch);
            auto offset=((uint64_t(b)*s.extent+group)*meta.capacity+pos)*d*2;
            if(!vector_.Copy(keys+uint64_t(pos)*d*2,p(1)+offset,d) ||
                !vector_.Copy(values+uint64_t(pos)*d*2,p(2)+offset,d))return false;
          }
        }
        Acquire(scratch);
        unsigned end=(qb+1)*s.query_rows;if(end>meta.seq*s.group)end=meta.seq*s.group;
        for(unsigned qi=qb*s.query_rows;qi<end;++qi) {
          unsigned pos=qi/s.group,q=qi%s.group;
          auto source=p(0)+((uint64_t(b)*meta.seq+pos)*packed+uint64_t(group)*(s.group+2)*d+q*d)*2;
          if(!Rotate(source,p(5),p(3)+uint64_t(pos)*d*2,p(4)+uint64_t(pos)*d*2,scratch,scratch+512,d,s.epsilon))return false;
          Acquire(scratch);
          auto output=p(7)+(((uint64_t(b)*meta.seq+pos)*s.extent+group)*s.group+q)*d*2;
          if(!Attend(scratch,keys,values,output,scratch,pos+1,d,s.variant))return false;
        }
        return true;
      }
      unsigned blocks=(meta.capacity+s.kv_block-1)/s.kv_block;
      unsigned block=task%blocks,group=(task/blocks)%s.extent,b=task/(blocks*s.extent),d=s.width;
      unsigned begin=block*s.kv_block,end=past+1;
      if(end>begin+s.kv_block)end=begin+s.kv_block;
      if(end>meta.capacity || b>=meta.batch)return false;
      uint64_t packed=uint64_t(s.extent)*(s.group+2)*d;
      auto qkv=p(0)+(uint64_t(b)*packed+uint64_t(group)*(s.group+2)*d)*2;
      auto key=p(1)+(uint64_t(b)*s.extent+group)*meta.capacity*d*2;
      auto value=p(2)+(uint64_t(b)*s.extent+group)*meta.capacity*d*2;
      auto cos=p(3)+uint64_t(past)*d*2,sin=p(4)+uint64_t(past)*d*2;
      // Only the task owning the current cache block appends this K/V row.
      if(begin<=past && past<begin+s.kv_block) {
        if(!Rotate(qkv+s.group*d*2,p(6),cos,sin,key+uint64_t(past)*d*2,scratch+512,d,s.epsilon) ||
            !vector_.Copy(qkv+(s.group+1)*d*2,value+uint64_t(past)*d*2,d))return false;
        Acquire(key);
      }
      for(unsigned q=0;q<s.group;++q) {
        auto po=p(8)+(((uint64_t(b)*s.extent+group)*blocks+block)*s.group+q)*d*4;
        auto lse=p(9)+(((uint64_t(b)*s.extent+group)*blocks+block)*s.group+q)*4;
        if(!Rotate(qkv+q*d*2,p(5),cos,sin,scratch,scratch+512,d,s.epsilon))return false;
        Acquire(scratch);
        if(blocks==1) {
          if(!Attend(scratch,key,value,p(7)+((uint64_t(b)*s.extent+group)*s.group+q)*d*2,scratch,end,d,s.variant))return false;
          continue;
        }
        unsigned chunks=0;
        for(unsigned k=begin;k<end;k+=32,++chunks) {
          unsigned n=end-k<32?end-k:32;
          if(!vector_.AttentionChunk(scratch,key+uint64_t(k)*d*2,value+uint64_t(k)*d*2,
              scratch+1024+chunks*d*4,scratch+4096+chunks*8,n,d,
              d==128?0.08838834764831845f:0.125f))return false;
        }
        Acquire(scratch);
        if(!vector_.ServingPartial(scratch+1024,scratch+4096,po,lse,chunks,d))return false;
      }
      return true;
    }
    if(s.kind==14) {
      unsigned group=task%s.extent,b=task/s.extent,blocks=(meta.capacity+s.kv_block-1)/s.kv_block;
      if(b>=meta.batch)return false;
      auto partial=p(0)+(uint64_t(b)*s.extent+group)*blocks*s.group*s.width*4;
      auto lse=p(1)+(uint64_t(b)*s.extent+group)*blocks*s.group*4;
      for(unsigned q=0;q<s.group;++q)
        if(!vector_.ServingMerge(partial,lse,p(2)+((uint64_t(b)*s.extent+group)*s.group+q)*s.width*2,
            blocks,q,s.group,s.width))return false;
      return true;
    }
    if(s.kind==15) {
      if(task>=meta.batch || !s.width)return false;
      AscendC::GlobalTensor<float> values;AscendC::GlobalTensor<uint32_t> indices;
      values.SetGlobalBuffer((__gm__ float*)p(0)+uint64_t(task)*s.width);
      indices.SetGlobalBuffer((__gm__ uint32_t*)p(1)+uint64_t(task)*s.width);
      float maximum=-3.4e38f;uint32_t index=UINT32_MAX;
      for(unsigned i=0;i<s.width;++i) {
        float v=values.GetValue(i);auto id=indices.GetValue(i);
        if(id>=s.extent || !(v==v))return false;
        if(v>maximum || (v==maximum && id<index)){maximum=v;index=id;}
      }
      if(index==UINT32_MAX || past+meta.seq>=meta.capacity)return false;
      WriteControl((__gm__ uint32_t*)p(2)+uint64_t(task)*meta.capacity+past+meta.seq,index);return true;
    }
    return false;
  }
};
} // namespace tilemega::ascend
