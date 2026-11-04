// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <tilemega/Solver/Sm90NativeCtg.h>
#include <tilemega/Solver/Sm90NativeMoeDomain.h>
#include <tilemega/Analysis/ISLContext.h>
#include <fstream>
#include <iostream>
#include <set>
using namespace tilemega;
using namespace tilemega::solver;

int main(int argc,char** argv) {try {
  if(argc!=4)throw std::invalid_argument("usage: tilemega-native-domains target.json cells.json output.jsonl");
  analysis::IslContext isl;
  auto target=TargetSpec::FromJson(argv[1]);
  if(std::ifstream(argv[3]))throw std::invalid_argument("retain previous domain output; choose a new path");
  std::ofstream output(argv[3]);if(!output)throw std::runtime_error("cannot open domain output");
  auto cells=json::ParseFile(argv[2]);
  for(auto const& cell:cells.AsArray("cells")) {
    auto seed=Sm90DenseCoordinates::FromJson(json::ParseFile(cell.At("seed").AsString("seed")));
    mlir::MLIRContext context;
    auto imported=ImportSm90Dense(cell.At("export").AsString("export"),seed,context);
    auto work=Sm90DenseWorkloadFromPlan(imported.plan,cell.At("batch").AsNumber("batch"));
    Sm90DenseSearchSpace space{work.shapes,work.batch,work.head_dim,work.queries,target,work.hkv};
    space.moe=work.moe;space.experts=work.experts;space.top_k=work.top_k;
    space.qk_norm_row=work.qk_norm_row;space.tp=work.tp;
    json::Array domains;
    for(int residency:{1,2}) {
      for(auto family:seed.Active()) {
        json::Array options;std::set<int> partitions;
        for(auto g:space.FamilyCandidates(family,residency)) {
          int n=work.shapes[family][0],k=work.shapes[family][1],tiles=n/g.tile_n;
          double active=family>=6?Sm90NativeExpectedActiveExperts(work.experts,work.top_k,work.batch):1;
          int tasks=family>=6?Sm90NativeExpectedExpertTasks(work.experts,work.top_k,work.batch,tiles):tiles;
          double rows=family==6?work.batch*double(work.top_k)/active:
              family==7?(work.batch<=8?8:16):work.batch;
          auto c=seed;c.gemms[family]=g;
          // Same bytes as Sm90DenseCostModel::Gemm, before the service-rate model.
          // Predictive task cardinality uses the approved finite-domain rounding.
          double weights=double(tasks)*g.tile_n*k*2,activations=double(tasks)*rows*k*2;
          options.emplace_back(json::Object{{"geometry",c.ToJson().At("gemms").At(kSm90DenseFamilies[family])},
              {"tasks",tasks},{"expected_active_experts",active},{"rows_per_task",rows},
              {"weight_bytes",weights},{"activation_bytes",activations},
              {"traffic_bytes",weights+activations},
              {"traffic_task_window",tasks*2>=space.Grid(residency) && tasks<=space.Grid(residency)*2}});
          partitions.insert(g.tile_n);
        }
        json::Array theta;for(int tn:partitions)theta.emplace_back(tn);
        domains.emplace_back(json::Object{{"family",kSm90DenseFamilies[family]},
            {"residency",residency},{"grid",space.Grid(residency)},
            {"partition_axis","TN"},{"partition_values",theta},{"geometries",options}});
      }
      json::Array options;std::set<int> partitions;
      for(auto a:space.AttentionCandidates(residency)) {
        auto c=seed;c.attention=a;
        options.emplace_back(c.ToJson().At("attention"));partitions.insert(a.chunks);
      }
      json::Array theta;for(int chunks:partitions)theta.emplace_back(chunks);
      domains.emplace_back(json::Object{{"family","attention"},{"residency",residency},
          {"grid",space.Grid(residency)},{"partition_axis","chunks"},
          {"partition_values",theta},{"geometries",options}});
    }
    Sm90NativeCtgCache cache;
    auto graph=InstantiateSm90NativeCtg(imported,seed,work.batch,context,cache,space.Grid(seed.residency));
    json::Array nodes,edges;
    for(auto const& task:graph.spaces) {
      std::string family;
      if(task.role==frontend::OpRole::kServingFusedAttention)family="attention";
      else if(task.role==frontend::OpRole::kServingMerge)family="attention_merge";
      else if(task.role==frontend::OpRole::kMoERouting)family="route";
      else if(task.role==frontend::OpRole::kMoECombine)family="combine";
      else if(task.role==frontend::OpRole::kServingArgmax)family="argmax";
      else if(task.role==frontend::OpRole::kEmbedding)family="embedding";
      else {
        auto const& stage=imported.plan.stages.at(task.stage);
        family=kSm90DenseFamilies[Sm90DenseFamily(imported.plan,imported.plan.gemms.at(stage.gemm))];
      }
      nodes.emplace_back(json::Object{{"name",task.name},{"family",family},
          {"stage",task.stage},{"tasks",task.count},{"ownership",task.ownership.ToString()}});
    }
    for(auto const& edge:graph.edges)edges.emplace_back(json::Object{
        {"producer",edge.producer},{"consumer",edge.consumer}});
    json::Value row=json::Object{{"input",cell},{"layers",work.layers},{"domains",domains},
        {"seed",seed.ToJson()},{"seed_graph",json::Object{{"spaces",nodes},{"edges",edges}}},
        {"approximation",graph.approximation},
        {"scope","rule-pruned family domains; partition axes exclude body-only TK/warps/slots; seed CTG topology"}};
    output<<row.Dump(0);output.flush();
    std::cout<<cell.At("id").AsString("id")<<" domains="<<domains.size()<<'\n'<<std::flush;
  }
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
