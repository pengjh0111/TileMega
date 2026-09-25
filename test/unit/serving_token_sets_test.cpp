// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Analysis/ISLContext.h>

#include <cassert>
#include <iostream>

using namespace tilemega::analysis;

int main() {
  IslContext isl;
  auto B = ClosedForm::Symbol("B");
  auto past = ClosedForm::Symbol("past");
  auto S = ClosedForm::Constant(64);
  TensorSpace tokens{"serving.tokens", {{"b", B},
      {"pos", ClosedForm::Constant(1088)}}, ""};
  TensorSpace hidden{"serving.hidden", {{"b", B}}, ""};
  SemanticOp embed;
  embed.name = "embed";
  embed.kind = OperatorKind::kPointwise;
  embed.dtype = ScalarType::kBF16;
  embed.domain = {{"b", B}};
  embed.result = hidden;
  embed.result_map.results = {IndexResult::Dim("b")};
  SemanticOperand read;
  read.tensor = tokens;
  read.map.results = {IndexResult::Dim("b"), IndexResult::Affine({}, past)};
  embed.operands.push_back(read);
  embed.element_reads.push_back({read.tensor, read.map, {}});

  SemanticOp argmax;
  argmax.name = "argmax";
  argmax.kind = OperatorKind::kPointwise;
  argmax.dtype = ScalarType::kBF16;
  argmax.domain = {{"b", B}};
  argmax.result = tokens;
  argmax.result.axes[1].origin = past + S;
  argmax.result.axes[1].extent = ClosedForm::Constant(1);
  argmax.result_map.results = {IndexResult::Dim("b"), IndexResult::Affine({})};
  argmax.result_effect.state_object = "serving.tokens";
  SemanticOperand consume;
  consume.producer = "embed";
  consume.tensor = hidden;
  consume.map.results = {IndexResult::Dim("b")};
  argmax.operands.push_back(consume);

  DramFloorOptions options;
  options.dram_gbps = 981.582923;
  options.tc_gflops = 179997;
  options.element_bytes["serving.tokens"] = 4;
  SemanticGraph graph;
  graph.ops = {embed, argmax};
  auto floor = DeriveDramFloor(graph, options, {});
  ParamBinding point;
  point.Bind("B", 2).Bind("past", 64);
  auto value = floor.Evaluate(point);
  assert(value.read_bytes == 8);
  assert(value.write_bytes == 8);
  auto const& footprint = floor.tensors.at("serving.tokens");
  assert(footprint.reads.IsSubset(
      footprint.reads.Subtract(footprint.writes)));
  std::cout << "TOKENS_DISJOINT batch=2 past=64 seq=64 read_bytes="
            << value.read_bytes << " write_bytes=" << value.write_bytes << '\n';
}
