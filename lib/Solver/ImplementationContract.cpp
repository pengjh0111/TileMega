// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ImplementationContract.h>

#include <sstream>
#include <stdexcept>

namespace tilemega::solver {
namespace {

std::string Fail(std::string* error, std::string message) {
  if (error) *error = message;
  return message;
}

/// The single task coordinate an access base advances along, or "" when the
/// axis is covered whole. More than one coordinate on one axis is a packed
/// index; the contract cannot describe it, so it is reported rather than
/// silently reduced to its first term.
bool DrivingCoordinate(analysis::AffineExpr const& base, std::string* out,
                       std::string* error) {
  std::vector<std::string> coordinates = base.Coordinates();
  if (coordinates.size() > 1)
    return Fail(error, "axis is driven by more than one task coordinate"), false;
  *out = coordinates.empty() ? std::string() : coordinates.front();
  return true;
}

}  // namespace

BackendTraits ImplementationContract::DeclaredTraits() const {
  BackendTraits traits;
  traits.tile_m = tile_m;
  traits.tile_n = tile_n;
  traits.tile_k = tile_k;
  traits.stages = stages;
  traits.threads = threads;
  traits.smem_bytes = smem_bytes;
  traits.cluster = cluster;
  traits.alignment = alignment;
  traits.arch_sm = arch_required;
  traits.shape_legal = SimtF32ShapeLegal(tile_m, tile_n, tile_k, stages);
  return traits;
}

bool VerifyTraits(ImplementationContract const& impl, std::string* error) {
  if (impl.backend != kSimtF32Backend)
    return Fail(error, "unknown backend '" + impl.backend + "'"), false;
  BackendTraits expected =
      SimtF32Traits(impl.tile_m, impl.tile_n, impl.tile_k, impl.stages);
  if (!expected.shape_legal)
    return Fail(error, "illegal tile shape for " + impl.backend), false;
  std::ostringstream out;
  if (impl.threads != expected.threads)
    out << "threads " << impl.threads << " != " << expected.threads << "; ";
  if (impl.smem_bytes != expected.smem_bytes)
    out << "smem " << impl.smem_bytes << " != " << expected.smem_bytes << "; ";
  if (impl.alignment.a != expected.alignment.a ||
      impl.alignment.b != expected.alignment.b)
    out << "alignment does not match the backend's; ";
  if (impl.arch_required < expected.arch_sm)
    out << "arch_required " << impl.arch_required << " < " << expected.arch_sm
        << "; ";
  if (impl.cluster.size() < 1)
    out << "cluster shape must be positive; ";
  std::string message = out.str();
  if (message.empty()) return true;
  return Fail(error, "declared traits contradict " + impl.backend + ": " +
                         message.substr(0, message.size() - 2)),
         false;
}

bool VerifyTarget(ImplementationContract const& impl, TargetSpec const& target,
                  std::string* error) {
  BackendCandidate candidate(impl.DeclaredTraits());
  if (!candidate.isLegal(target))
    return Fail(error, "implementation is not legal on " + target.arch_tag), false;
  if (impl.regs_est && *impl.regs_est > 0 && target.res.regs_per_sm > 0) {
    long needed = static_cast<long>(*impl.regs_est) * impl.threads;
    if (needed > target.res.regs_per_sm)
      return Fail(error, "declared register estimate does not fit one SM"), false;
  }
  return true;
}

bool LowerAccess(std::vector<analysis::AccessRelation> const& derived,
                 analysis::ParamBinding const& theta,
                 analysis::ParamBinding const& g,
                 std::vector<OperandContract>* out, std::string* error) {
  out->clear();
  for (auto const& relation : derived) {
    OperandContract operand;
    operand.name = relation.is_write ? "C" : relation.producer;
    for (std::size_t axis = 0; axis < relation.index.size(); ++axis) {
      std::string coordinate;
      std::string reason;
      if (!DrivingCoordinate(relation.index[axis].base, &coordinate, &reason))
        return Fail(error, operand.name + " axis " + std::to_string(axis) +
                               ": " + reason),
               false;
      operand.coordinate.push_back(coordinate);
      try {
        operand.span.push_back(
            static_cast<int>(relation.index[axis].span.Eval(theta, g)));
      } catch (std::exception const& failure) {
        return Fail(error, operand.name + " axis " + std::to_string(axis) +
                               ": cannot evaluate the derived span " +
                               relation.index[axis].span.ToString() + " (" +
                               failure.what() + ")"),
               false;
      }
    }
    out->push_back(std::move(operand));
  }
  return true;
}

bool VerifyAccessAgainst(ImplementationContract const& impl,
                         std::vector<OperandContract> const& derived,
                         std::string* error) {
  if (impl.operands.size() != derived.size())
    return Fail(error, "contract declares " +
                           std::to_string(impl.operands.size()) +
                           " operands, the task space has " +
                           std::to_string(derived.size())),
           false;
  for (std::size_t i = 0; i < derived.size(); ++i) {
    OperandContract const& declared = impl.operands[i];
    OperandContract const& actual = derived[i];
    std::string where = impl.name + " operand " + declared.name;
    if (declared.coordinate.size() != actual.coordinate.size() ||
        declared.span.size() != actual.coordinate.size())
      return Fail(error, where + ": declared rank " +
                             std::to_string(declared.coordinate.size()) +
                             " != tensor rank " +
                             std::to_string(actual.coordinate.size())),
             false;
    for (std::size_t axis = 0; axis < actual.coordinate.size(); ++axis) {
      // Transposition shows up here and only here: a transposed operand can
      // have identical tile extents (a square tile) while the coordinate
      // driving each axis is swapped.
      if (declared.coordinate[axis] != actual.coordinate[axis])
        return Fail(error, where + " axis " + std::to_string(axis) +
                               ": declared coordinate '" +
                               declared.coordinate[axis] +
                               "', task space indexes it by '" +
                               (actual.coordinate[axis].empty()
                                    ? "<whole axis>"
                                    : actual.coordinate[axis]) +
                               "'"),
               false;
      if (declared.span[axis] != actual.span[axis])
        return Fail(error, where + " axis " + std::to_string(axis) +
                               ": declared span " +
                               std::to_string(declared.span[axis]) +
                               ", task space touches " +
                               std::to_string(actual.span[axis])),
               false;
    }
  }
  return true;
}

bool VerifyAccess(ImplementationContract const& impl,
                  std::vector<analysis::AccessRelation> const& derived,
                  analysis::ParamBinding const& theta,
                  analysis::ParamBinding const& g, std::string* error) {
  std::vector<OperandContract> lowered;
  if (!LowerAccess(derived, theta, g, &lowered, error)) return false;
  return VerifyAccessAgainst(impl, lowered, error);
}

bool VerifyContract(ImplementationContract const& impl, TargetSpec const& target,
                    std::vector<analysis::AccessRelation> const& derived,
                    analysis::ParamBinding const& theta,
                    analysis::ParamBinding const& g, std::string* error) {
  return VerifyTraits(impl, error) && VerifyTarget(impl, target, error) &&
         VerifyAccess(impl, derived, theta, g, error);
}

bool SelectImplementation(CandidateGenerator const& generator,
                          GemmProblem const& problem, std::string name,
                          std::string task,
                          std::vector<analysis::AccessRelation> const& derived,
                          analysis::ParamBinding const& theta,
                          TileSymbols const& symbols, TileCandidate* chosen_g,
                          ImplementationContract* chosen_impl,
                          analysis::ParamBinding* chosen_binding) {
  std::vector<BackendCandidate> legal = generator.Enumerate();
  std::vector<BackendCandidate> best = generator.TopK(legal, {problem}, 1);
  if (best.empty()) return false;
  BackendTraits const& traits = best.front().traits();

  // The granularity and the implementation are one decision: `g` is what the
  // derived access maps are parameterized in, so choosing a tile *is*
  // choosing the spans the contract will have to declare.
  analysis::ParamBinding binding;
  binding.Bind(symbols.m, traits.tile_m);
  binding.Bind(symbols.n, traits.tile_n);
  binding.Bind(symbols.k, traits.tile_k);

  ImplementationContract impl;
  impl.name = std::move(name);
  impl.task = std::move(task);
  impl.backend = kSimtF32Backend;
  impl.tile_m = traits.tile_m;
  impl.tile_n = traits.tile_n;
  impl.tile_k = traits.tile_k;
  impl.cluster = traits.cluster;
  impl.stages = traits.stages;
  impl.threads = traits.threads;
  impl.smem_bytes = traits.smem_bytes;
  impl.alignment = traits.alignment;
  impl.arch_required = traits.arch_sm;
  if (!LowerAccess(derived, theta, binding, &impl.operands, nullptr))
    return false;

  if (chosen_g) *chosen_g = {traits.tile_m, traits.tile_n, traits.tile_k,
                             traits.stages};
  if (chosen_binding) *chosen_binding = binding;
  *chosen_impl = std::move(impl);
  return true;
}

}  // namespace tilemega::solver
