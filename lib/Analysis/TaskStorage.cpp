// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/TaskStorage.h>
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace tilemega::analysis {
namespace {
ClosedForm C(long x) { return ClosedForm::Constant(x); }
struct Storage {
  TensorSpace logical;
  TileStoragePartition spec;
  ClosedForm parts, tile;
  IndexResult writer;
  bool whole=false;
};
bool SameShape(TensorSpace const& a, TensorSpace const& b) {
  if(a.axes.size()!=b.axes.size() || a.layout_id!=b.layout_id)return false;
  for(unsigned i=0;i<a.axes.size();++i)
    if(a.axes[i].extent.ToString()!=b.axes[i].extent.ToString() ||
       a.axes[i].origin.ToString()!=b.axes[i].origin.ToString())return false;
  return true;
}
}
void ValidateTileStorage(SemanticOp const& op) {
  if(op.tile_storage.empty() && op.tile_storage_reads.empty())return;
  if(!op.exact_task_access)
    throw std::invalid_argument("tile storage requires exact task ownership");
  std::set<std::string> tensors;
  for(auto const& spec:op.tile_storage) {
    if(spec.tensor.empty() || !tensors.insert(spec.tensor).second || spec.tensor==op.result.name)
      throw std::invalid_argument("tile storage must name a distinct side tensor");
    auto axis=std::find_if(op.task_space.axes.begin(),op.task_space.axes.end(),
        [&](auto const& a){return a.name==spec.owner_axis;});
    if(axis==op.task_space.axes.end() || op.task_map.results.size()!=op.task_space.axes.size())
      throw std::invalid_argument("tile storage names an unknown ownership axis");
    auto const& index=op.task_map.results[axis-op.task_space.axes.begin()];
    if(index.kind!=IndexResult::Kind::kAffine || !index.outer_divisor.IsLiteral(1))
      throw std::invalid_argument("tile storage requires an affine ownership coordinate");
    for(auto const& term:index.terms)
      if(!op.Dim(term.dim) || op.Dim(term.dim)->type!=IteratorType::kParallel)
        throw std::invalid_argument("tile storage ownership depends on a reduction coordinate");
    bool found=false;
    for(auto const& write:op.additional_writes)if(write.tensor.name==spec.tensor) {
      found=true;
      if(spec.tensor_axis>write.tensor.axes.size() || write.map.results.size()!=write.tensor.axes.size())
        throw std::invalid_argument("invalid tile storage insertion axis");
    }
    if(!found)throw std::invalid_argument("tile storage has no declared side write");
  }
  tensors.clear();
  for(auto const& selection:op.tile_storage_reads) {
    auto reduction=op.Dim(selection.reduction_dim),segment=op.Dim(selection.segment_dim);
    if(!tensors.insert(selection.tensor).second || !reduction || !segment ||
       reduction->type!=IteratorType::kReduction || segment->type!=IteratorType::kParallel ||
       !reduction->origin.IsLiteral(0) || !segment->origin.IsLiteral(0) ||
       !selection.segment_extent.IsConstant() || selection.segment_extent.Eval({},{})<=0)
      throw std::invalid_argument("invalid segmented tile storage selection");
    if(std::none_of(op.operands.begin(),op.operands.end(),
        [&](auto const& operand){return operand.tensor.name==selection.tensor;}))
      throw std::invalid_argument("tile storage selection has no semantic operand");
  }
}
SemanticGraph MaterializeTaskStorage(SemanticGraph const& graph, Granularity const& g) {
  std::map<std::string,Storage> storage;
  for(auto const& op:graph.ops) {
    ValidateTileStorage(op);
    for(auto const& spec:op.tile_storage) {
      auto axis=std::find_if(op.task_space.axes.begin(),op.task_space.axes.end(),
          [&](auto const& a){return a.name==spec.owner_axis;});
      auto write=std::find_if(op.additional_writes.begin(),op.additional_writes.end(),
          [&](auto const& w){return w.tensor.name==spec.tensor;});
      Storage s;s.logical=write->tensor;s.spec=spec;
      s.whole=!g.TileOf(op.name,spec.owner_axis,&s.tile);
      if(s.whole) {
        s.tile=axis->extent;s.parts=C(1);s.writer=IndexResult::Affine({},C(0));
      }else {
        if(!s.tile.IsConstant() || s.tile.Eval({},{})<=0)
          throw std::invalid_argument("tile storage requires a concrete positive ownership tile");
        s.parts=axis->extent.CeilDiv(s.tile);
        s.writer=op.task_map.results[axis-op.task_space.axes.begin()];
        s.writer.outer_divisor=s.tile;
      }
      if(!storage.emplace(spec.tensor,std::move(s)).second)
        throw std::invalid_argument("tile storage has multiple producers");
    }
  }
  if(storage.empty()) {
    for(auto const& op:graph.ops)if(!op.tile_storage_reads.empty())
      throw std::invalid_argument("tile storage selection has no producer");
    return graph;
  }
  auto result=graph;
  for(auto& op:result.ops) {
    std::map<std::string,TileStorageSelection const*> selections;
    for(auto const& selection:op.tile_storage_reads) {
      auto found=storage.find(selection.tensor);
      if(found==storage.end())throw std::invalid_argument("tile storage selection has no producer");
      auto const& s=found->second;
      for(auto& dim:op.domain)if(dim.name==selection.reduction_dim)dim.extent=s.parts;
      if(!s.whole) {
        auto upper=IndexResult::Affine({{selection.reduction_dim,s.tile,C(1)},
            {selection.segment_dim,C(-1)*selection.segment_extent,C(1)}},s.tile+C(-1));
        auto lower=IndexResult::Affine({{selection.segment_dim,selection.segment_extent,C(1)},
            {selection.reduction_dim,C(-1)*s.tile,C(1)}},selection.segment_extent+C(-1));
        op.domain_nonnegative.push_back(std::move(upper));
        op.domain_nonnegative.push_back(std::move(lower));
      }
      selections.emplace(selection.tensor,&selection);
    }
    auto expand=[&](TensorSpace& tensor,IndexingMap& map,bool write) {
      auto found=storage.find(tensor.name);if(found==storage.end())return;
      auto const& s=found->second;
      if(!SameShape(tensor,s.logical) || map.results.size()!=tensor.axes.size())
        throw std::invalid_argument("tile storage references disagree on logical shape");
      std::string axis="tile_part";
      while(std::any_of(tensor.axes.begin(),tensor.axes.end(),
          [&](auto const& a){return a.name==axis;}))axis+="_";
      tensor.axes.insert(tensor.axes.begin()+s.spec.tensor_axis,{axis,s.parts});
      auto index=write?s.writer:IndexResult::FullRange();
      if(auto selected=selections.find(tensor.name);selected!=selections.end())
        index=IndexResult::Dim(selected->second->reduction_dim);
      map.results.insert(map.results.begin()+s.spec.tensor_axis,std::move(index));
    };
    expand(op.result,op.result_map,true);
    for(auto& write:op.additional_writes)expand(write.tensor,write.map,true);
    for(auto& operand:op.operands)expand(operand.tensor,operand.map,false);
    for(auto& operand:op.epilogue_operands)expand(operand.tensor,operand.map,false);
    for(auto& read:op.element_reads)expand(read.tensor,read.map,false);
    op.tile_storage.clear();op.tile_storage_reads.clear();
  }
  return result;
}
} // namespace tilemega::analysis
