// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/Semantics.h>

#include <sstream>
#include <utility>

namespace tilemega::analysis {

std::string ToString(IteratorType type) {
  return type == IteratorType::kParallel ? "parallel" : "reduction";
}

std::string ToString(ScalarType type) {
  return type == ScalarType::kBF16 ? "bf16" : "f32";
}

std::string ToString(EffectKind kind) {
  switch (kind) {
    case EffectKind::kRead: return "read";
    case EffectKind::kWrite: return "write";
    case EffectKind::kReadWrite: return "readwrite";
  }
  return "read";
}

IndexResult IndexResult::Dim(std::string name, ClosedForm coefficient,
                             ClosedForm group) {
  IndexResult result;
  result.terms.push_back({std::move(name), std::move(coefficient),
                          std::move(group)});
  return result;
}

IndexResult IndexResult::Affine(std::vector<Term> terms, ClosedForm offset) {
  IndexResult result;
  result.terms = std::move(terms);
  result.offset = std::move(offset);
  return result;
}

IndexResult IndexResult::FullRange(ClosedForm offset) {
  IndexResult result;
  result.kind = Kind::kFullRange;
  result.offset = std::move(offset);
  return result;
}

IndexResult IndexResult::Broadcast(ClosedForm span) {
  IndexResult result;
  result.kind = Kind::kBroadcast;
  result.span = std::move(span);
  return result;
}

IndexResult IndexResult::DataDependent() {
  IndexResult result;
  result.kind = Kind::kDataDependent;
  return result;
}

std::string IndexResult::Serialize() const {
  std::ostringstream out;
  switch (kind) {
    case Kind::kAffine: {
      if (terms.empty()) out << offset.ToString();
      for (std::size_t i = 0; i < terms.size(); ++i) {
        if (i) out << " + ";
        out << terms[i].coefficient.ToString() << "*";
        if (terms[i].group.IsLiteral(1)) out << terms[i].dim;
        else out << "floordiv(" << terms[i].dim << ", "
                 << terms[i].group.ToString() << ")";
      }
      if (!terms.empty() && !offset.IsLiteral(0))
        out << " + " << offset.ToString();
      break;
    }
    case Kind::kFullRange:
      out << "full[" << offset.ToString() << "]";
      break;
    case Kind::kBroadcast:
      out << "broadcast[" << span.ToString() << "]";
      break;
    case Kind::kDataDependent:
      out << "data_dependent";
      break;
  }
  return out.str();
}

std::string IndexingMap::Serialize() const {
  std::ostringstream out;
  out << "(";
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (i) out << ", ";
    out << results[i].Serialize();
  }
  out << ")";
  return out.str();
}

std::string MemoryEffect::Serialize() const {
  std::ostringstream out;
  out << ToString(kind);
  if (!alias_set.empty()) out << " alias=" << alias_set;
  if (!state_object.empty()) out << " state=" << state_object;
  return out.str();
}

std::string ReductionSemantics::Serialize() const {
  if (!splittable) return "none";
  std::ostringstream out;
  out << "split(dim=" << dim << ", op=" << reduction_operator
      << ", partial=" << partial_tensor << ", combiner=" << combiner
      << ", ownership=[";
  for (std::size_t i = 0; i < ownership.size(); ++i) {
    if (i) out << ",";
    out << ownership[i];
  }
  out << "])";
  return out.str();
}

namespace {

std::string SerializeTensor(TensorSpace const& tensor) {
  std::ostringstream out;
  out << tensor.name << "<";
  for (std::size_t i = 0; i < tensor.axes.size(); ++i) {
    if (i) out << ", ";
    out << tensor.axes[i].name << ":" << tensor.axes[i].extent.ToString();
    if (!tensor.axes[i].origin.IsLiteral(0))
      out << "@" << tensor.axes[i].origin.ToString();
    if (tensor.axes[i].runtime) out << "!";
  }
  out << ">";
  if (!tensor.layout_id.empty()) out << "[layout=" << tensor.layout_id << "]";
  return out.str();
}

}  // namespace

IterationDim const* SemanticOp::Dim(std::string const& name) const {
  for (auto const& dim : domain)
    if (dim.name == name) return &dim;
  return nullptr;
}

void SetRotationElementReads(SemanticOp& op, TensorSpace frequency, ClosedForm head_width) {
  if (op.operands.size()!=1 || op.operands[0].map.results.size()!=2)
    throw std::invalid_argument("rotation element reads require a rank-two source");
  auto const& source=op.operands[0];
  auto const& column=source.map.results[1];
  if (column.terms.size()!=1 || !column.terms[0].coefficient.IsLiteral(1) ||
      !column.terms[0].group.IsLiteral(1))
    throw std::invalid_argument("rotation source must expose its element column");
  auto dim=column.terms[0].dim;
  auto one=ClosedForm::Constant(1),two=ClosedForm::Constant(2),minus=ClosedForm::Constant(-1);
  auto half=head_width.FloorDiv(two);
  auto partner=source.map;
  // Swap half-heads without treating the discontinuous permutation as an
  // interval: h + D/2 - D*floor(h/(D/2)) + 2D*floor(h/D).
  partner.results[1]=IndexResult::Affine({{dim,one,one},{dim,minus*head_width,half},
                                        {dim,two*head_width,head_width}},half);
  IndexingMap freq{{IndexResult::Affine({{dim,one,one},{dim,minus*half,half}})}};
  op.element_reads={{source.tensor,source.map,{}},{source.tensor,partner,{}},
                    {std::move(frequency),std::move(freq),{}}};
}

void SetCausalAttentionReads(SemanticOp& op, ClosedForm head_width,
                            ClosedForm head_group, ClosedForm past) {
  if (op.operands.size()!=3 || op.result_map.results.size()!=2 || op.reduction.dim.empty())
    throw std::invalid_argument("causal attention requires Q/K/V and a reduction axis");
  auto q=op.operands[0]; auto k=op.operands[1]; auto v=op.operands[2];
  auto const& output=op.result_map.results;
  if (output[0].terms.size()!=1 || output[1].terms.size()!=1)
    throw std::invalid_argument("causal attention requires direct token/head coordinates");
  auto token=output[0].terms[0].dim,head=output[1].terms[0].dim;
  auto one=ClosedForm::Constant(1),minus=ClosedForm::Constant(-1);
  // Flattened element h = head*D+d maps to floor(head/G)*D+d, not floor(h/G).
  auto grouped=IndexResult::Affine({{head,head_width,head_width*head_group},
                                    {head,one,one},{head,minus*head_width,head_width}});
  k.map.results[1]=v.map.results[1]=grouped;
  auto causal=IndexResult::Affine({{token,one,one},{op.reduction.dim,minus,one}},past);
  // PV still reads all V positions, including positions whose probability is
  // zero; only QK's global K loads are guarded by the causal predicate.
  op.element_reads={{q.tensor,q.map,{}},{k.tensor,k.map,{causal}},{v.tensor,v.map,{}}};
}

std::string SemanticOp::Serialize() const {
  std::ostringstream out;
  out << "op " << name << " kind=" << ToString(kind)
      << " dtype=" << ToString(dtype)
      << (generic ? " generic" : "");
  if (!arithmetic.empty()) out << " arithmetic=" << arithmetic;
  for (auto const& read:element_reads) {
    out << "\n  element_read " << SerializeTensor(read.tensor) << ' ' << read.map.Serialize();
    for (auto const& predicate:read.nonnegative)
      out << " where " << predicate.Serialize() << ">=0";
  }
  out << "\n  domain";
  for (auto const& dim : domain) {
    out << " " << dim.name << ":" << ToString(dim.type) << "["
        << dim.origin.ToString() << ", " << dim.extent.ToString() << ")";
    if (dim.runtime) out << "!";
  }
  out << "\n  result " << SerializeTensor(result) << " "
      << result_map.Serialize() << " " << result_effect.Serialize() << "\n";
  for (auto const& operand : operands)
    out << "  operand " << (operand.producer.empty() ? "-" : operand.producer)
        << " " << SerializeTensor(operand.tensor) << " "
        << operand.map.Serialize() << " " << operand.effect.Serialize() << "\n";
  out << "  reduction " << reduction.Serialize() << "\n";
  return out.str();
}

SemanticOp const* SemanticGraph::Find(std::string const& name) const {
  for (auto const& op : ops)
    if (op.name == name) return &op;
  return nullptr;
}

std::string SemanticGraph::Serialize() const {
  std::ostringstream out;
  for (auto const& op : ops) out << op.Serialize();
  return out.str();
}

SemanticOp GenericSemantics(std::string name, TensorSpace result,
                            std::vector<SemanticOperand> operands) {
  SemanticOp op;
  op.name = std::move(name);
  op.kind = OperatorKind::kPointwise;
  op.generic = true;
  op.result = std::move(result);
  op.result_effect.kind = EffectKind::kWrite;
  for (auto const& axis : op.result.axes) {
    IterationDim dim;
    dim.name = axis.name;
    dim.extent = axis.extent;
    dim.origin = axis.origin;
    dim.runtime = axis.runtime;
    op.domain.push_back(dim);
    op.result_map.results.push_back(IndexResult::Dim(axis.name));
  }
  op.operands = std::move(operands);
  for (auto& operand : op.operands) {
    operand.map.results.clear();
    for (auto const& axis : operand.tensor.axes) {
      (void)axis;
      operand.map.results.push_back(IndexResult::FullRange());
    }
    operand.effect.kind = EffectKind::kRead;
  }
  return op;
}

}  // namespace tilemega::analysis
