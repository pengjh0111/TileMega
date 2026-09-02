// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CGContract.h>

#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;

namespace tilemega::dialect {
namespace {

bool Fail(std::string* error, std::string message) {
  if (error) *error = std::move(message);
  return false;
}

}  // namespace

AccessMapAttr emitOperandContract(Builder& builder,
                                  solver::OperandContract const& operand) {
  SmallVector<Attribute> coordinates;
  SmallVector<int64_t> spans;
  for (auto const& name : operand.coordinate)
    coordinates.push_back(builder.getStringAttr(name));
  for (int span : operand.span) spans.push_back(span);
  return AccessMapAttr::get(
      builder.getContext(),
      builder.getDictionaryAttr(
          {builder.getNamedAttr("operand", builder.getStringAttr(operand.name)),
           builder.getNamedAttr("coordinates", builder.getArrayAttr(coordinates)),
           builder.getNamedAttr("spans", builder.getDenseI64ArrayAttr(spans))}));
}

ArrayAttr emitOperandContracts(
    Builder& builder, std::vector<solver::OperandContract> const& operands) {
  SmallVector<Attribute> entries;
  for (auto const& operand : operands)
    entries.push_back(emitOperandContract(builder, operand));
  return builder.getArrayAttr(entries);
}

bool emitIndexMap(Builder& builder,
                  std::vector<analysis::AccessRelation> const& derived,
                  analysis::ParamBinding const& theta,
                  analysis::ParamBinding const& g, ArrayAttr* out,
                  std::string* error) {
  std::vector<solver::OperandContract> lowered;
  if (!solver::LowerAccess(derived, theta, g, &lowered, error)) return false;
  *out = emitOperandContracts(builder, lowered);
  return true;
}

bool readOperandContracts(ArrayAttr array,
                          std::vector<solver::OperandContract>* out,
                          std::string* error) {
  out->clear();
  for (Attribute entry : array) {
    auto access = dyn_cast<AccessMapAttr>(entry);
    if (!access) return Fail(error, "entry is not an access_map");
    DictionaryAttr fields = access.getFields();
    auto name = fields.getAs<StringAttr>("operand");
    auto coordinates = fields.getAs<ArrayAttr>("coordinates");
    auto spans = fields.getAs<DenseI64ArrayAttr>("spans");
    if (!name || !coordinates || !spans)
      return Fail(error, "access_map needs operand, coordinates and spans");
    if (coordinates.size() != static_cast<std::size_t>(spans.size()))
      return Fail(error, "operand '" + name.getValue().str() +
                             "' has a different number of coordinates and spans");
    solver::OperandContract operand;
    operand.name = name.getValue().str();
    for (Attribute coordinate : coordinates) {
      auto text = dyn_cast<StringAttr>(coordinate);
      if (!text) return Fail(error, "coordinate is not a string");
      operand.coordinate.push_back(text.getValue().str());
    }
    for (int64_t span : spans.asArrayRef())
      operand.span.push_back(static_cast<int>(span));
    out->push_back(std::move(operand));
  }
  return true;
}

bool readImplementation(ImplementationOp op, solver::ImplementationContract* out,
                        std::string* error) {
  ArrayRef<int64_t> tile = op.getTile();
  ArrayRef<int64_t> cluster = op.getCluster();
  ArrayRef<int64_t> alignment = op.getAlignment();
  if (tile.size() != 3) return Fail(error, "tile must be [m, n, k]");
  if (cluster.size() != 3) return Fail(error, "cluster must be [m, n, k]");
  if (alignment.size() != 2) return Fail(error, "alignment must be [a, b]");
  solver::ImplementationContract impl;
  impl.name = op.getSymName().str();
  impl.task = op.getTask().str();
  impl.backend = op.getBackend().str();
  impl.tile_m = static_cast<int>(tile[0]);
  impl.tile_n = static_cast<int>(tile[1]);
  impl.tile_k = static_cast<int>(tile[2]);
  impl.cluster = {static_cast<int>(cluster[0]), static_cast<int>(cluster[1]),
                  static_cast<int>(cluster[2])};
  impl.stages = static_cast<int>(op.getStages());
  impl.threads = static_cast<int>(op.getThreads());
  impl.smem_bytes = static_cast<int>(op.getSmemBytes());
  if (auto registers = op.getRegsEst())
    impl.regs_est = static_cast<int>(*registers);
  impl.alignment = {static_cast<int>(alignment[0]),
                    static_cast<int>(alignment[1])};
  impl.arch_required = static_cast<int>(op.getArchRequired());
  if (!readOperandContracts(op.getAccess(), &impl.operands, error)) return false;
  *out = std::move(impl);
  return true;
}

ImplementationOp emitImplementation(OpBuilder& builder, Location loc,
                                    solver::ImplementationContract const& impl) {
  OperationState state(loc, ImplementationOp::getOperationName());
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(impl.name));
  state.addAttribute("task", FlatSymbolRefAttr::get(builder.getContext(),
                                                    impl.task));
  state.addAttribute("backend", builder.getStringAttr(impl.backend));
  state.addAttribute("tile", builder.getDenseI64ArrayAttr(
                                 {impl.tile_m, impl.tile_n, impl.tile_k}));
  state.addAttribute("cluster",
                     builder.getDenseI64ArrayAttr({impl.cluster.m, impl.cluster.n,
                                                   impl.cluster.k}));
  state.addAttribute("stages", builder.getI64IntegerAttr(impl.stages));
  state.addAttribute("threads", builder.getI64IntegerAttr(impl.threads));
  state.addAttribute("smem_bytes", builder.getI64IntegerAttr(impl.smem_bytes));
  if (impl.regs_est)
    state.addAttribute("regs_est", builder.getI64IntegerAttr(*impl.regs_est));
  state.addAttribute("alignment", builder.getDenseI64ArrayAttr(
                                      {impl.alignment.a, impl.alignment.b}));
  state.addAttribute("arch_required",
                     builder.getI64IntegerAttr(impl.arch_required));
  state.addAttribute("access", emitOperandContracts(builder, impl.operands));
  return cast<ImplementationOp>(builder.create(state));
}

}  // namespace tilemega::dialect
