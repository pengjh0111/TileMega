// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/BindingRequestTraffic.h>
#include <tilemega/Analysis/ISLContext.h>
#include <map>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
BindingRequestTraffic DeriveBindingRequestTraffic(analysis::OperatorNode const& task,
    analysis::DramFloor const& floor,analysis::ParamBinding const& known) {
  analysis::IslReferenceAudit audit(__func__);
  using namespace analysis;
  if(!task.element_access)throw std::invalid_argument("binding traffic needs exact L-task accesses");
  auto const& access=*task.element_access;
  auto const& sem=access.semantic;
  BindingRequestTraffic result;
  // A scalar zero is not an ISL task-coordinate polynomial. Start each sum
  // with its first actual fiber instead of adding that scalar seed.
  std::vector<QuasiPolynomial> read_bytes,write_bytes,external_reads,external_writes;
  struct Read {CouplingRelation issued,physical;bool requests=false;};
  std::map<std::pair<std::string,std::set<std::string>>,Read> reads;
  auto append=[&](TensorSpace const& tensor,IndexingMap const& map,
                  std::vector<IndexResult> const& predicates) {
    std::set<std::string> sources;
    for(auto const& index:map.results)if(!index.request_dims.empty())sources.insert(index.binding_source);
    auto& row=reads[{tensor.name,sources}];
    bool requests=HasBindingRequests(map);
    auto physical=ProjectTaskRead(sem,task,access.partition,tensor,map,predicates,known);
    auto issued=requests?ProjectTaskRequests(sem,task,access.partition,tensor,map,predicates,known):physical;
    row.physical=row.physical.Union(physical);row.issued=row.issued.Union(issued);row.requests=requests;
  };
  if(sem.element_reads.empty())for(auto const& input:sem.operands)append(input.tensor,input.map,{});
  else for(auto const& input:sem.element_reads)append(input.tensor,input.map,input.nonnegative);
  std::set<std::string> live;
  for(auto const& [key,row]:reads) {
    auto found=floor.tensors.find(key.first);
    if(found==floor.tensors.end() || found->second.element_bytes<=0)
      throw std::invalid_argument("binding traffic lacks a typed physical tensor footprint");
    auto const& tensor=found->second;
    auto writes=tensor.writes.BindParams(known),external=tensor.no_producer.BindParams(known);
    auto count=row.issued.BoundTaskCard().Scale(tensor.element_bytes);
    read_bytes.push_back(count);
    if(tensor.binding_producer) {
      if(!row.physical.Image().IsSubset(tensor.binding_producer->envelope.BindParams(known)))
        throw std::invalid_argument("binding read exceeds its produced capacity");
      live.insert(key.first);
    } else if(row.requests) {
      auto image=row.physical.Image();
      if(writes.empty() || image.IsSubset(external))
        external_reads.push_back(count);
      else if(image.IsSubset(writes))live.insert(key.first);
      else throw std::invalid_argument("mixed indirect read provenance requires the runtime binding image");
    } else {
      auto outside=external.empty()?row.physical.Subtract(row.physical):
          row.physical.ApplyRange(external.ImageIdentity());
      external_reads.push_back(outside.BoundTaskCard().Scale(tensor.element_bytes));
      if(!row.physical.IsSubset(outside))live.insert(key.first);
    }
  }
  for(auto const& name:live) {
    auto const& tensor=floor.tensors.at(name);
    auto const& allocated=tensor.binding_producer?tensor.binding_producer->envelope:tensor.writes;
    auto elements=allocated.BindParams(known).ImageCard().Eval({});
    if(elements<0 || std::uint64_t(elements)>
        (UINT64_MAX-result.produced_live_bytes)/unsigned(tensor.element_bytes))
      throw std::overflow_error("binding traffic live footprint overflows");
    result.produced_live_bytes+=std::uint64_t(elements)*tensor.element_bytes;
  }
  result.produced_tensors=std::move(live);
  auto store=[&](TensorSpace const& tensor,IndexingMap const& map,
                 std::vector<IndexResult> const& predicates) {
    auto found=floor.tensors.find(tensor.name);
    if(found==floor.tensors.end() || found->second.element_bytes<=0)
      throw std::invalid_argument("binding traffic lacks a typed store footprint");
    auto const& footprint=found->second;
    auto physical=ProjectTaskWrite(sem,task,access.partition,tensor,map,predicates,known);
    bool requests=HasBindingRequests(map);
    auto issued=requests?ProjectTaskRequests(sem,task,access.partition,tensor,map,predicates,known):physical;
    auto count=issued.BoundTaskCard().Scale(footprint.element_bytes);
    write_bytes.push_back(count);
    auto external=footprint.external_writes.BindParams(known);
    if(external.empty())return;
    auto outside=physical.ApplyRange(external.ImageIdentity());
    if(requests) {
      if(physical.IsSubset(outside))external_writes.push_back(count);
      else if(!outside.IsSubset(physical.Subtract(physical)))
        throw std::invalid_argument("mixed indirect store provenance requires the runtime binding image");
    } else external_writes.push_back(outside.BoundTaskCard().Scale(footprint.element_bytes));
  };
  store(sem.result,sem.result_map,{});
  for(auto const& side:sem.additional_writes)store(side.tensor,side.map,side.nonnegative);
  result.read_bytes=QuasiPolynomial::Sum(read_bytes);
  result.write_bytes=QuasiPolynomial::Sum(write_bytes);
  result.no_producer_read_bytes=QuasiPolynomial::Sum(external_reads);
  result.external_write_bytes=QuasiPolynomial::Sum(external_writes);
  return result;
}
} // namespace tilemega::solver
