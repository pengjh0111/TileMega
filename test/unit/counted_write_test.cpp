// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CountedDependencyForm.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>

namespace tilemega::tests::counted_write_test {
using namespace analysis;
ClosedForm F(long x){return ClosedForm::Constant(x);}
int TestCountedWrite(int,char**) {
  IslContext isl;
  for(int tokens:{1,2,5,17})for(int block:{1,2,4})
    for(int consumer_rows:{1,2,4,16})for(int channels_per_task:{4,8}) {
    int topk=2,channels=9,slots=tokens*topk;
    SemanticOp scatter;scatter.name="down";scatter.exact_task_access=true;
    IterationDim virtual_row;virtual_row.name="v";virtual_row.extent=ClosedForm::Symbol("live");
    virtual_row.runtime=true;virtual_row.capacity=F(slots);
    virtual_row.binding_source="rows";virtual_row.binding_requirement="tensor_values";
    scatter.domain={virtual_row,{"c",F(channels)}};
    scatter.result={"partial",{{"t",F(tokens)},{"r",F(topk)},{"c",F(channels)}}};
    scatter.result_map.results={IndexResult::DataDependent("rows"),
        IndexResult::DataDependent("rows"),IndexResult::Dim("c")};
    scatter.task_space={"partial",{{"v",F(slots)},{"c",F(channels)}}};
    scatter.task_map.results={IndexResult::Dim("v"),IndexResult::Dim("c")};
    SemanticOp combine;combine.name="combine";combine.exact_task_access=true;
    combine.domain={{"t",F(tokens)},{"r",F(topk),F(0),IteratorType::kReduction},{"c",F(channels)}};
    combine.result={"output",{{"t",F(tokens)},{"c",F(channels)}}};
    combine.result_map.results={IndexResult::Dim("t"),IndexResult::Dim("c")};
    combine.task_space=combine.result;combine.task_map=combine.result_map;
    IndexingMap partial;partial.results={IndexResult::Dim("t"),IndexResult::Dim("r"),IndexResult::Dim("c")};
    combine.operands.push_back({scatter.name,scatter.result,partial,{}});
    Granularity geometry;geometry.Tile("down","v",F(block)).Tile("down","c",F(channels_per_task));
    geometry.Tile("combine","t",F(consumer_rows)).Tile("combine","c",F(channels_per_task));
    auto graph=Instantiate({{scatter,combine}},geometry);
    auto const& producer=graph.nodes[0];auto const& consumer=graph.nodes[1];
    auto edges=CouplingDerivation{}.Derive(graph,{});
    assert(edges.size()==1 && !edges[0].exact && edges[0].tier==Tier::kDataDependent);
    assert(edges[0].attributes.runtime_requirement==RuntimeRequirement::kTensorValues);
    assert(edges[0].attributes.countability==Countability::kUncountable);
    for(auto const& [to,from]:edges[0].C.Points())assert(to.back()==from.back());
    auto reads=ProjectTaskRead(combine,consumer,consumer.element_access->partition,
        scatter.result,partial,{});
    auto contract=BindCountedTaskDependency(consumer,reads,{0,1},"rows");
    auto aligned=BindAlignedCountedScatterDependency(producer,consumer,"partial",{0,1},"rows");
    assert(aligned.expected==contract.expected && aligned.target_units==contract.target_units);
    auto renamed=producer;renamed.name+=".combine";
    auto split_aligned=BindAlignedCountedScatterDependency(renamed,consumer,"partial",{0,1},"rows");
    assert(split_aligned.expected==contract.expected && split_aligned.target_units==contract.target_units);
    int ntiles=(channels+channels_per_task-1)/channels_per_task;
    for(unsigned target=0;target<contract.expected.size();++target) {
      int begin=(target/ntiles)*consumer_rows;
      assert(contract.expected[target]==unsigned(std::min(consumer_rows,tokens-begin)*topk));
    }
    assert(std::accumulate(contract.expected.begin(),contract.expected.end(),0u)==unsigned(slots*ntiles));
    auto rejected=[](auto action){bool yes=false;try{action();}catch(std::exception const&){yes=true;}assert(yes);};
    rejected([&]{BindCountedTaskDependency(consumer,reads,{0,0},"rows");});
    rejected([&]{BindCountedTaskDependency(consumer,reads,{2,0},"rows");});
    rejected([&]{BindCountedTaskDependency(consumer,reads,{3},"rows");});
    rejected([&]{BindCountedTaskDependency(consumer,reads,{0,1},"");});
    rejected([&]{BindAlignedCountedScatterDependency(producer,consumer,"partial",{0,1},"other");});
    rejected([&]{BindAlignedCountedScatterDependency(producer,consumer,"missing",{0,1},"rows");});
    rejected([&]{BindAlignedCountedScatterDependency(producer,consumer,"partial",{0},"rows");});
    auto misaligned_geometry=geometry;
    misaligned_geometry.Tile("down","c",F(channels_per_task==4?8:4));
    auto misaligned=Instantiate({{scatter,combine}},misaligned_geometry);
    rejected([&]{BindAlignedCountedScatterDependency(misaligned.nodes[0],misaligned.nodes[1],"partial",{0,1},"rows");});
    if(tokens!=2 || block!=2 || consumer_rows!=1 || channels_per_task!=4)continue;
    std::vector<int> rows(slots);std::iota(rows.begin(),rows.end(),0);
    bool one_producer=false,two_producers=false;
    do {
      std::map<std::pair<int,int>,unsigned> arrivals;
      std::ostringstream relation;relation<<"{ ";bool first=true;
      for(int v=0;v<slots;++v)for(int n=0;n<ntiles;++n) {
        int token=rows[v]/topk,target=token*ntiles+n,source=(v/block)*ntiles+n;
        ++arrivals[{target,source}];
        if(!first)relation<<"; ";first=false;
        relation<<"[t,n] -> [v,c] : t="<<token<<" and n="<<n<<" and v="<<(v/block)<<" and c="<<n;
      }
      relation<<" }";auto actual=CouplingRelation::FromIslText(relation.str());
      assert(Contains(edges[0].C,actual));
      std::vector<unsigned> weights(contract.expected.size()),tasks(weights);
      for(auto const& [pair,weight]:arrivals){weights[pair.first]+=weight;++tasks[pair.first];}
      assert(weights==contract.expected);
      for(auto count:tasks){one_producer|=count==1;two_producers|=count==2;}
    }while(std::next_permutation(rows.begin(),rows.end()));
    assert(one_producer && two_producers);
    auto invalid=scatter.result_map;invalid.results[0].binding_source.clear();
    rejected([&]{ProjectTaskWrite(scatter,producer,producer.element_access->partition,scatter.result,invalid,{});});
  }
  std::cout<<"Counted writes: 96 geometries, tail thresholds, I2 and 24 weighted binding permutations PASS\n";
  return 0;
}
} // namespace tilemega::tests::counted_write_test
