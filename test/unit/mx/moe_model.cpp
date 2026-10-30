// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/MxMoeModel.h>
#include <tilemega/Codegen/MxMoeProgramV2.h>
#include <tilemega/Solver/MxMoeSelectionV2.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <llvm/Support/JSON.h>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace tilemega::frontend;
using namespace tilemega::codegen;
namespace {
ModelPlan Fixture(unsigned batch,unsigned layers,bool merge,bool fp32) {
  ModelPlan p;p.dm=p.serving=true;p.serving_seq=1;p.serving_capacity=1088;
  p.norm_epsilon=1e-6;p.token_id_bits=32;
  auto buffer=[&](std::string name,std::string dtype="bf16",bool weight=false,std::string recipe="") {
    unsigned id=p.buffers.size();PlanBuffer b;b.name=std::move(name);b.dtype=std::move(dtype);b.pack_json=std::move(recipe);
    if(weight)b.source=PlanBuffer::Source::kWeight;
    p.buffers.push_back(b);return id;
  };
  auto stage=[&](PlanTaskKind kind) -> PlanStage& {p.stages.emplace_back();p.stages.back().kind=kind;return p.stages.back();};
  auto gemm=[&](unsigned n,unsigned k,unsigned a,unsigned w,unsigned out)->unsigned {
    unsigned id=p.gemms.size();PlanGemm g;g.n=n;g.k=k;g.a=a;g.b=w;g.c=a;g.d=out;p.gemms.push_back(g);
    stage(PlanTaskKind::kGemm).gemm=id;return id;
  };
  auto fold=[](std::string leaf){return "{\"kind\":\"fold_rmsnorm\",\"norm\":\"gamma\",\"source\":"+leaf+"}";};
  unsigned tokens=buffer("tokens","i32"),embed=buffer("embed","bf16",true),cos=buffer("cos"),sin=buffer("sin"),current=buffer("x0");p.buffers[tokens].role="external";p.buffers[cos].role=p.buffers[sin].role="external";
  auto& embedding=stage(PlanTaskKind::kEmbedding);embedding.operands[0]=tokens;embedding.operands[1]=embed;embedding.operands[2]=current;embedding.width=2048;embedding.extent=151936;
  for(unsigned l=0;l<layers;++l) {
    std::string const prefix="layer"+std::to_string(l)+".";
    unsigned qkv_out=buffer(prefix+"qkv"),context=buffer(prefix+"context"),xprime=buffer(prefix+"xprime"),statistics=buffer(prefix+"stats","f32");
    unsigned qweight=buffer(prefix+"qweight","bf16",true,fold(R"({"kind":"qkv_group_interleave","hkv":4,"qperkv":8,"head_dim":128})"));
    unsigned q=gemm(5120,2048,current,qweight,qkv_out);p.gemms[q].norm_ss=statistics;
    unsigned kc=buffer(prefix+"k"),vc=buffer(prefix+"v"),qn=buffer(prefix+"qn","bf16",true),kn=buffer(prefix+"kn","bf16",true);p.buffers[qn].constant=p.buffers[kn].constant=128;p.buffers[kc].role=p.buffers[vc].role="external";
    auto& att=stage(PlanTaskKind::kFusedAttention);att.extent=4;att.width=128;att.group=8;
    att.operands[0]=qkv_out;att.operands[1]=kc;att.operands[2]=vc;att.operands[3]=cos;att.operands[4]=sin;att.operands[5]=qn;att.operands[6]=kn;att.operands[7]=context;
    if(merge){auto& a=stage(PlanTaskKind::kAttentionMerge);a.extent=4;a.width=128;a.group=8;a.operands[2]=context;}
    unsigned ow=buffer(prefix+"ow","bf16",true);unsigned o=gemm(2048,4096,context,ow,xprime);p.gemms[o].c=current;p.gemms[o].beta=1;p.gemms[o].epilogue=PlanGemm::Epilogue::kResidual;
    unsigned logits=buffer(prefix+"logits"),partial=buffer(prefix+"partial","f32"),indices=buffer(prefix+"indices","i32"),routing=buffer(prefix+"weights",fp32?"f32":"bf16"),bindings=buffer(prefix+"bindings","i32"),rows=buffer(prefix+"rows","i32");
    unsigned rw=buffer(prefix+"rw","bf16",true,fold(R"({"kind":"alias","source":"router"})"));unsigned router=gemm(128,2048,xprime,rw,logits);
    auto& chain=p.gemms[router].chain;chain.count=1;chain.operations[0].kind=DmEpilogueKind::kDeferredRMSNorm;chain.operations[0].parameter[0]=statistics;chain.operations[0].output_rounding=DmRounding::kBF16;
    chain.side_count=1;chain.side[0].kind=DmSideOutputKind::kTopKPartial;chain.side[0].buffer=partial;chain.side[0].auxiliary=indices;chain.side[0].count=8;
    auto& top=stage(PlanTaskKind::kMoETopK);top.moe.step=DmMoeStep::kSelectAndDispatch;top.moe.experts=128;top.moe.top_k=8;top.moe.row_capacity=batch*8;top.moe.router_gemm=router;top.operands[0]=partial;top.operands[1]=indices;top.operands[3]=routing;top.operands[4]=bindings;top.operands[5]=rows;
    auto expert_recipe=[](char const* part){return std::string("{\"kind\":\"expert_stack\",\"part\":\"")+part+"\",\"prefix\":\"experts.\",\"experts\":128,\"hidden\":2048,\"intermediate\":768,\"u\":16}";};
    unsigned guw=buffer(prefix+"guw","bf16",true,fold(expert_recipe("gate_up"))),dw=buffer(prefix+"dw","bf16",true,expert_recipe("down")),h=buffer(prefix+"h"),partials=buffer(prefix+"partials"),output=buffer(prefix+"out");
    unsigned gu=gemm(1536,2048,xprime,guw,h);auto& g=p.gemms[gu];g.access.a=DmAAccess::kRowGather;g.access.b=DmBAccess::kExpertIndirect;g.access.binding=bindings;g.access.rows=rows;g.access.expert_stride=1536*2048;
    g.chain=p.gemms[router].chain;g.chain.side_count=0;g.chain.count=2;auto& gate=g.chain.operations[1];gate.kind=DmEpilogueKind::kGatePair;gate.gate=DmGatePair::kSwiGLU;gate.unit=16;gate.input_rounding=gate.output_rounding=DmRounding::kBF16;
    unsigned down=gemm(2048,768,h,dw,partials);auto& d=p.gemms[down];d.access.b=DmBAccess::kExpertIndirect;d.access.binding=bindings;d.access.expert_stride=2048*768;d.access.write.kind=DmWriteKind::kRowScatter;d.access.write.rows=rows;
    auto& combine=stage(PlanTaskKind::kMoECombine);combine.moe.step=DmMoeStep::kCombine;combine.moe.experts=128;combine.moe.top_k=8;combine.extent=2048;combine.operands[0]=partials;combine.operands[1]=routing;combine.operands[2]=xprime;combine.operands[3]=output;current=output;
  }
  unsigned hw=buffer("head.weight","bf16",true,fold(R"({"kind":"alias","source":"head"})")),values=buffer("head.values","f32"),indices=buffer("head.indices","i32");unsigned head=gemm(151936,2048,current,hw,values);p.gemms[head].epilogue=PlanGemm::Epilogue::kArgmaxPartial;
  auto& arg=stage(PlanTaskKind::kArgmaxReduce);arg.operands[0]=values;arg.operands[1]=indices;arg.operands[2]=tokens;return p;
}
}
int main(int argc,char** argv) {
 try {
  unsigned positive=0,negative=0,emitted=0,configuration_negative=0,buffer_negative=0;
  for(unsigned b:{1u,2u,4u,8u,16u})for(unsigned layers:{1u,2u,24u})for(bool merge:{false,true})for(bool fp32:{false,true}) {
    auto p=Fixture(b,layers,merge,fp32);auto m=AnalyzeMxMoe(p);
    assert(m.batch==b&&m.layers.size()==layers&&m.hidden==2048&&m.hkv==4&&m.queries==8&&m.routing_weights_fp32==fp32);++positive;
    auto prepared=PrepareMxMoeBuffersV2(p,{16,32,64,64,128,128});
    assert(prepared.size()==m.external_buffers.size());unsigned wrapped=0;
    for(unsigned index=0;index<prepared.size();++index) {
      auto const& native=prepared[index];auto const& original=m.external_buffers[index];
      assert(native.original_index==original.original_index&&native.buffer.name==original.buffer.name&&native.buffer.dtype==original.buffer.dtype&&native.buffer.constant==original.buffer.constant&&native.buffer.per_batch==original.buffer.per_batch);
      if(native.slot==MxMoeSlotV2::kQkv||native.slot==MxMoeSlotV2::kO||native.slot==MxMoeSlotV2::kRouter||native.slot==MxMoeSlotV2::kGateUp||native.slot==MxMoeSlotV2::kDown||native.slot==MxMoeSlotV2::kHead) {
        auto recipe=llvm::json::parse(native.buffer.pack_json);assert(recipe);auto* tile=recipe->getAsObject();assert(tile&&tile->getString("kind")=="tile_pages"&&tile->getString("layout")=="xcore_mma"&&tile->getInteger("tile_k")==32);++wrapped;
      } else assert(native.buffer.pack_json==original.buffer.pack_json);
    }
    assert(wrapped==5*layers+1);
    for(auto invalid:{MxMoeWeightTilesV2{16,32,64,16,128,128},MxMoeWeightTilesV2{16,32,64,64,128,0}}) {
      bool rejected=false;try{PrepareMxMoeBuffersV2(p,invalid);}catch(std::invalid_argument const&){rejected=true;}assert(rejected);++buffer_negative;
    }
    MxMoeConfigV2 config;
    config.threads=512;config.b_min=4;config.attn_prefetch=0;
    config.qkv={64,1,1,false,1,-1};config.o=config.qkv;
    config.router={32,1,1,true,1,-1};config.gate_up={64,2,1,false,0,-1};
    config.down={64,2,1,false,1,-1};config.head={32,8,1,false,1,-1};
    CheckMxMoeConfigV2(p,config);
    if (b == 1 && layers == 2 && !merge && fp32) {
      using namespace tilemega::solver;
      std::array<std::vector<MxMoeStageCostV2>, 7> costs;
      std::array<MxDecodeClassV2, 6> classes{config.qkv, config.o, config.router,
          config.gate_up, config.down, config.head};
      for (unsigned i = 0; i < classes.size(); ++i) {
        classes[i].r_next = 120;
        MxMoeStageCostV2 row; row.ns = 100 + i; row.gemm = classes[i];
        if (i == 2) row.dispatch_gate_tn = row.dispatch_down_tn = 64;
        costs[i].push_back(row);
      }
      for (int bmin : {8, 1, 4, 2}) {
        MxMoeStageCostV2 row; row.ns = 200 + bmin; row.b_min = bmin;
        costs[6].push_back(row);
      }
      // A cheaper GU is not compatible with the only measured router context.
      auto other = costs[3][0]; other.ns = 1; other.gemm.tn = 32;
      costs[3].push_back(other);
      auto ranked = RankMxMoeProgramsV2(p, 512, costs);
      assert(ranked.size() == 4);
      for (unsigned i = 0; i < ranked.size(); ++i) {
        auto const& choice = ranked[i];
        assert(choice.config.gate_up.tn == 64 && choice.config.down.tn == 64);
        assert(choice.config.threads == 512 && choice.source_rows[3] == 0);
        assert(choice.config.b_min == (1 << i));
        double expected = costs[5][0].ns;
        for (unsigned stage : {0, 1, 2, 3, 4, 6})
          expected += 2 * costs[stage][choice.source_rows[stage]].ns;
        assert(choice.predicted_ns == expected);
        CheckMxMoeConfigV2(p, choice.config);
      }
      assert(RankMxMoeProgramsV2(p, 512, costs, 0).empty());
      // The context-specific router row makes the cheaper GU eligible.
      auto router = costs[2][0]; router.ns += 1; router.dispatch_gate_tn = 32;
      costs[2].push_back(router);
      ranked = RankMxMoeProgramsV2(p, 512, costs);
      assert(ranked.front().config.gate_up.tn == 32 && ranked.front().source_rows[2] == 1);
      std::mt19937 random(2030);
      for (int trial = 0; trial < 40; ++trial) {
        auto varied = costs;
        for (auto& list : varied) for (auto& row : list) row.ns = 1 + random() % 5;
        std::array<std::vector<std::size_t>, 7> order;
        for (unsigned stage = 0; stage < varied.size(); ++stage) {
          for (unsigned i = 0; i < varied[stage].size(); ++i) order[stage].push_back(i);
          std::stable_sort(order[stage].begin(), order[stage].end(), [&](auto a, auto b) {
            return varied[stage][a].ns < varied[stage][b].ns;
          });
        }
        std::vector<std::pair<double, MxMoeRanksV2>> expected;
        MxMoeRanksV2 ranks{};
        auto enumerate = [&](auto&& self, unsigned stage, double sum) -> void {
          if (stage == varied.size()) {
            auto const& router = varied[2][order[2][ranks[2]]];
            if (router.dispatch_gate_tn == varied[3][order[3][ranks[3]]].gemm.tn)
              expected.emplace_back(sum, ranks);
            return;
          }
          for (unsigned i = 0; i < order[stage].size(); ++i) {
            ranks[stage] = i;
            self(self, stage + 1, sum + varied[stage][order[stage][i]].ns * (stage == 5 ? 1 : 2));
          }
        };
        enumerate(enumerate, 0, 0);
        std::sort(expected.begin(), expected.end());
        auto actual = RankMxMoeProgramsV2(p, 512, varied);
        assert(actual.size() == 4);
        for (unsigned i = 0; i < actual.size(); ++i) {
          assert(actual[i].predicted_ns == expected[i].first);
          for (unsigned stage = 0; stage < varied.size(); ++stage)
            assert(actual[i].source_rows[stage] == order[stage][expected[i].second[stage]]);
        }
      }
      for (int mutation = 0; mutation < 8; ++mutation) {
        auto bad = costs;
        if (mutation == 0) bad[0][0].ns = 0;
        if (mutation == 1) bad[2][0].dispatch_gate_tn = 0;
        if (mutation == 2) bad[0][0].gemm.r_next = -1;
        if (mutation == 3) bad[3][0].gemm.prefetch = 1;
        if (mutation == 4) bad[6].push_back(bad[6][0]);
        if (mutation == 5) bad[0][0].dispatch_down_tn = 64;
        if (mutation == 6) bad[1].clear();
        if (mutation == 7) {
          auto duplicate = bad[0][0]; ++duplicate.gemm.r_next;
          bad[0].push_back(duplicate);
        }
        bool rejected = false;
        try { RankMxMoeProgramsV2(p, 512, bad); }
        catch (std::invalid_argument const&) { rejected = true; }
        assert(rejected);
      }
      auto incompatible = costs;
      incompatible[2].resize(1); incompatible[2][0].dispatch_down_tn = 128;
      assert(RankMxMoeProgramsV2(p, 512, incompatible).empty());
      {
        using tilemega::json::Array;
        using tilemega::json::Object;
        using tilemega::json::Value;
        // Complete synthetic geometry exercises serialization and source-index
        // mapping; these placeholders are never qualified measurements.
        std::vector<Value> aggregate;
        auto append = [&](std::string kind, Value key, int gu = 0, int dn = 0) {
          std::string implementation = kind == "moe_gate_up" || kind == "moe_down_combine" ? "moe_grouped" : kind;
          Value row = Object{{"kind", kind}, {"key", key}, {"median_ns", 100. + aggregate.size()},
              {"p90_ns", 101. + aggregate.size()}, {"spread", .01}, {"processes", 3},
              {"r_next", kind == "attention" ? 240 : 120},
              {"rounds", Array{Object{{"round", 1}}, Object{{"round", 2}}, Object{{"round", 3}}}},
              {"resources", Object{{"mtreg", 160}, {"local_bytes", 0}, {"static_smem_bytes", 0}, {"resident_ctas", 1}}},
              {"provenance", Object{{"executor", "mx_static_stage_bench"}, {"implementation", "mx5_" + implementation + "_v2"},
                  {"source_sha256", std::string(64, 'a')}, {"binary_sha256", std::string(64, 'b')}, {"base_head", "synthetic_cpu_contract"}}}};
          if (gu) row.Set("dispatch_context", Object{{"gate_up_tn", gu}, {"down_tn", dn}});
          aggregate.push_back(row);
        };
        constexpr std::array<char const*, 6> names{"qkv", "o", "router", "gate_up", "down", "head"};
        constexpr std::array<char const*, 6> kinds{"gemm", "gemm", "moe_router", "moe_gate_up", "moe_down_combine", "head"};
        for (unsigned stage = 0; stage < names.size(); ++stage) {
          auto shape = MxMoeGeometryV2(m, names[stage], b);
          for (auto const& item : MxMoeGemmDomainV2(m, names[stage], b)) {
            Value key = Object{{"n", shape.n}, {"k", shape.k}, {"b", int(b)}, {"threads", 512},
                {"tn", item.tn}, {"wn", item.wn}, {"depth", item.d}, {"prefetch", stage == 3 ? 0 : item.d}};
            if (stage != 5) { key.Set("op", shape.op); key.Set("norm", shape.norm); key.Set("part", int(!item.stream_k)); }
            if (shape.n_active) key.Set("n_active", shape.n_active);
            if (stage == 2) for (int gu : {32, 64, 128}) for (int dn : {16, 32, 64, 128}) append(kinds[stage], key, gu, dn);
            else append(kinds[stage], key);
          }
        }
        for (int bmin : {1, 2, 4, 8}) for (int length : {65, 256, 512, 768, 1088})
          append("attention", Object{{"dim", 128}, {"q", 8}, {"hkv", 4}, {"b", int(b)},
              {"length", length}, {"b_min", bmin}, {"qk_norm", 1}, {"capacity", 1088},
              {"threads", 512}, {"prefetch_kind", 0}});
        auto preview = MxMoeCandidatesJsonV2(p, 512, aggregate, {});
        assert(preview.At("coverage").At("missing").AsNumber("candidate number") == 0);
        assert(!preview.At("execution_authorized").AsBool("candidate flag") && !preview.At("file_qualification_implied").AsBool("candidate flag"));
        auto const& candidates = preview.At("candidates").AsArray("candidate array");
        assert(candidates.size() == 4);
        auto extended = MxMoeCandidatesJsonV2(p, 512, aggregate, {}, 8);
        auto const& diagnostic = extended.At("candidates").AsArray("candidate array");
        assert(diagnostic.size() == 8);
        for (std::size_t i = 0; i < candidates.size(); ++i)
          assert(diagnostic[i].Dump() == candidates[i].Dump());
        assert(!extended.At("execution_authorized").AsBool("candidate flag"));
        for (std::size_t invalid : {std::size_t(0), std::size_t(4097)}) {
          bool rejected_count = false;
          try { MxMoeCandidatesJsonV2(p, 512, aggregate, {}, invalid); }
          catch (std::invalid_argument const&) { rejected_count = true; }
          assert(rejected_count);
        }
        double previous = 0;
        for (auto const& candidate : candidates) {
          double prediction = candidate.At("predicted_tpot_ns").AsNumber("candidate number"), total = 0;
          assert(prediction >= previous); previous = prediction;
          auto const& evidence = candidate.At("evidence_row_indices");
          auto const& pin = candidate.At("pin").At("mx_moe_v2");
          for (char const* name : {"qkv", "o", "router", "gate_up", "down", "head", "attention"}) {
            auto const& indices = evidence.At(name).AsArray("candidate array");
            bool attention = std::string(name) == "attention", head = std::string(name) == "head";
            assert(indices.size() == (attention ? 5 : 1));
            total += candidate.At("predicted_stage_ns").At(name).AsNumber("candidate number") * (head ? 1 : layers);
            auto const& first = aggregate.at(std::size_t(indices[0].AsNumber("candidate number"))).At("key");
            if (!attention) assert(first.At("tn").AsNumber("candidate number") == pin.At(name).At("tn").AsNumber("candidate number"));
            else assert(first.At("b_min").AsNumber("candidate number") == pin.At("b_min").AsNumber("candidate number"));
          }
          assert(total == prediction && candidate.At("prompt_pair_required").AsBool("candidate flag"));
          auto const& router = aggregate.at(std::size_t(evidence.At("router").AsArray("candidate array")[0].AsNumber("candidate number")));
          assert(router.At("dispatch_context").At("gate_up_tn").AsNumber("candidate number") == pin.At("gate_up").At("tn").AsNumber("candidate number"));
        }
        aggregate.pop_back();
        bool rejected = false;
        try { MxMoeCandidatesJsonV2(p, 512, aggregate, {}); }
        catch (std::invalid_argument const&) { rejected = true; }
        assert(rejected);
      }
      std::cout << "PASS MoE ranked program assembly: dispatch contexts, layer costs, source rows\n";
    }
    std::string paired_buffers;
    for(bool prompt:{false,true}) {
      auto code=EmitMxMoeProgramV2(p,config,prompt);
      auto begin=code.find(" kBuffers[] = {");auto end=code.find("};",begin);assert(begin!=std::string::npos&&end!=std::string::npos);
      auto buffers=code.substr(begin,end-begin);
      if(prompt)assert(paired_buffers==buffers);else paired_buffers=buffers;
      assert(code.find("kIntermediate=768")!=std::string::npos);
      assert(code.find("MxRouterGemmV2<32,1,MxPart::kStreamK,Dispatch,1>")!=std::string::npos);
      assert(code.find("MxGroupedGemm<64,2,1,MxOpV2::kSwiGLU,true,Chain,MxNoUnitCompletionV2,0>")!=std::string::npos);
      assert(code.find("MxMoeDownCompletionV2<"+std::string(fp32?"true":"false")+">")!=std::string::npos);
      assert(code.find("QkvWeights(b,l+1==kLayers?0:l+1)")!=std::string::npos);
      assert((code.find("&& (past == 63)")!=std::string::npos)==prompt);
      assert(code.find("stage_descriptor")==std::string::npos);++emitted;
    }
    for(int mutant=0;mutant<4;++mutant) {
      auto invalid=config;
      if(mutant==0)invalid.gate_up.prefetch=1;
      if(mutant==1)invalid.down.stream_k=true;
      if(mutant==2)invalid.router.wn=2;
      if(mutant==3)invalid.head.wn=2;
      bool rejected=false;try{CheckMxMoeConfigV2(p,invalid);}catch(std::invalid_argument const&){rejected=true;}assert(rejected);++configuration_negative;
    }
    auto reject=[&](std::string const& location,std::function<void(ModelPlan&)> mutate){auto q=p;mutate(q);bool failed=false;try{AnalyzeMxMoe(q);}catch(std::invalid_argument const& e){failed=std::string(e.what()).find(location)!=std::string::npos;}if(!failed)throw std::runtime_error(std::string("missing rejection: ")+location);++negative;};
    reject("layer 0 qkv",[&](auto& q){q.gemms[q.stages[m.layers[0].qkv].gemm].n+=16;});
    reject("layer 0 dispatch",[&](auto& q){q.stages[m.layers[0].dispatch].moe.step=DmMoeStep::kSelect;});
    reject("layer 0 router recipe",[&](auto& q){q.buffers[q.gemms[q.stages[m.layers[0].router].gemm].b].pack_json="{";});
    reject("layer 0 router recipe",[&](auto& q){q.buffers[q.gemms[q.stages[m.layers[0].router].gemm].b].pack_json="{}";});
    reject("layer 0 expert recipes",[&](auto& q){auto& recipe=q.buffers[q.gemms[q.stages[m.layers[0].gate_up].gemm].b].pack_json;auto pos=recipe.find("gamma");recipe.replace(pos,5,"other");});
    reject("layer 0 gate_up",[&](auto& q){q.gemms[q.stages[m.layers[0].gate_up].gemm].chain.operations[1].output_rounding=DmRounding::kFP32;});
    reject("layer 0 experts",[&](auto& q){q.gemms[q.stages[m.layers[0].down].gemm].access.write.kind=DmWriteKind::kDense;});
    reject("layer "+std::to_string(layers)+" head",[&](auto& q){q.gemms[m.head_gemm].n+=16;});
  }
  if(argc==2){auto bridge=ReadExportBridge(argv[1]);for(unsigned b:{1u,2u,4u,8u,16u}){ServingOptions o;o.moe_batch=b;o.capacity=1088;o.phase=ServingOptions::Phase::kDecode;o.seq=1;o.deferred_norm=true;auto p=BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,o);auto m=AnalyzeMxMoe(p);assert(m.batch==b&&m.layers.size()==24&&m.external_buffers.size()==246);auto buffers=PrepareMxMoeBuffersV2(p,{16,32,64,64,128,128});assert(buffers.size()==246);unsigned unused=0;for(auto const& row:buffers)unused+=row.slot==MxMoeSlotV2::kUnused;assert(unused==25);++positive;}}
  std::cout<<"PASS portable MoE analysis positive="<<positive<<" negative="<<negative<<" emitted="<<emitted<<" configuration_negative="<<configuration_negative<<" buffer_negative="<<buffer_negative<<" GPU_calls=0\n";return 0;
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
