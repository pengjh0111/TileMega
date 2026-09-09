// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
//
// Part 1 acceptance: the derived metrics must be functions of the workload
// parameters, not integers measured at the bottom of the parameter range.
//
// Two claims are checked, both against the derivation itself rather than
// against a table typed in by hand:
//
//   (a) evaluating the symbolic metric at a point equals re-deriving the whole
//       coupling at that point, edge by edge and at five sequence lengths;
//   (b) the constant the frontend used to emit -- the metric derived with every
//       workload symbol pinned to its range minimum -- differs from the true
//       value, and the report says by how much.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace tilemega::analysis;

namespace {

/// Only what isl requires to be literal: a parametric divisor is rejected at
/// the C API, so tile sizes and the GQA group factor are bound.  S, L_s and
/// past stay free -- that is the whole point of the experiment.
ParamBinding FixedGranularity() {
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  return known;
}

ParamBinding At(long sequence, long past) {
  ParamBinding known = FixedGranularity();
  known.Bind("S", sequence);
  known.Bind("past", past);
  known.Bind("L_s", sequence + past);
  return known;
}

char const* Yes(bool value) { return value ? "ok" : "DIFFERS"; }

}  // namespace

int main() {
  tilemega::analysis::IslContext isl_context;
  DecoderShape shape;
  OperatorGraph const graph = LlamaDecoderLayer(shape);
  CouplingDerivation derivation;

  std::vector<CouplingEdge> const symbolic = derivation.Derive(graph, FixedGranularity());
  std::printf("EDGES %zu\n", symbolic.size());

  // (a) the symbolic metric, evaluated, against a fresh derivation at the point.
  std::vector<long> const sequences = {1, 4, 128, 512, 2048};
  std::printf("\n## symbolic-vs-pointwise\n");
  std::printf("seq\tedge\tsrc -> dst\twait\tfanout\tvolume\tcount\n");
  int checks = 0, agree = 0;
  for (long sequence : sequences) {
    // `past` follows the fixture convention: a prefill of `sequence` tokens on
    // top of a cache three rows deep, so L_s moves with S and neither is a
    // constant multiple of the other.
    ParamBinding const point = At(sequence, 3);
    std::vector<CouplingEdge> const here = derivation.Derive(graph, point);
    if (here.size() != symbolic.size()) {
      std::printf("SHAPE_MISMATCH seq=%ld %zu vs %zu\n", sequence, here.size(),
                  symbolic.size());
      return 1;
    }
    for (std::size_t i = 0; i < here.size(); ++i) {
      if (here[i].src.name != symbolic[i].src.name ||
          here[i].dst.name != symbolic[i].dst.name) {
        std::printf("ORDER_MISMATCH seq=%ld edge=%zu\n", sequence, i);
        return 1;
      }
      bool const w = symbolic[i].metrics.wait.SemanticallyEqual(
          here[i].metrics.wait, point);
      bool const f = symbolic[i].metrics.fanout.SemanticallyEqual(
          here[i].metrics.fanout, point);
      bool const v = symbolic[i].metrics.volume.SemanticallyEqual(
          here[i].metrics.volume, point);
      bool const c = symbolic[i].metrics.count.SemanticallyEqual(
          here[i].metrics.count, point);
      checks += 4;
      agree += int(w) + int(f) + int(v) + int(c);
      std::printf("%ld\t%zu\t%s -> %s\t%s\t%s\t%s\t%s\n", sequence, i + 1,
                  here[i].src.name.c_str(), here[i].dst.name.c_str(), Yes(w),
                  Yes(f), Yes(v), Yes(c));
    }
  }
  std::printf("\nAGREE %d/%d\n", agree, checks);

  // (b) what the old floor binding claimed, against what is true.  A metric
  // that is genuinely a function of the consumer coordinate has no single
  // value to quote; those edges are named rather than dropped, because "no
  // number here" is itself the reason a constant could not have been right.
  std::printf("\n## floor-binding error: metric derived at S=1,past=0 vs the "
              "truth at S=4096\n");
  std::vector<CouplingEdge> const floor = derivation.Derive(graph, At(1, 0));
  ParamBinding const big = At(4096, 0);
  std::printf("edge\tsrc -> dst\tmetric\tfloor\ttrue\tratio\n");
  int wrong = 0, positional = 0, same = 0;
  auto compare = [&](std::size_t i, char const* name,
                     QuasiPolynomial const& floor_metric,
                     QuasiPolynomial const& true_metric) {
    long floor_value = 0, true_value = 0;
    try {
      floor_value = floor_metric.Eval(At(1, 0));
      true_value = true_metric.Eval(big);
    } catch (std::exception const&) {
      ++positional;
      std::printf("%zu\t%s -> %s\t%s\tposition-dependent\tposition-dependent\t-\n",
                  i + 1, floor[i].src.name.c_str(), floor[i].dst.name.c_str(),
                  name);
      return;
    }
    if (floor_value == true_value) { ++same; return; }
    ++wrong;
    std::printf("%zu\t%s -> %s\t%s\t%ld\t%ld\t%.4gx\n", i + 1,
                floor[i].src.name.c_str(), floor[i].dst.name.c_str(), name,
                floor_value, true_value,
                floor_value == 0 ? 0.0
                                 : double(true_value) / double(floor_value));
  };
  for (std::size_t i = 0; i < floor.size(); ++i) {
    compare(i, "wait", floor[i].metrics.wait, symbolic[i].metrics.wait);
    compare(i, "fanout", floor[i].metrics.fanout, symbolic[i].metrics.fanout);
    compare(i, "count", floor[i].metrics.count, symbolic[i].metrics.count);
  }
  std::printf("\nFLOOR wrong=%d unchanged=%d position_dependent=%d\n", wrong,
              same, positional);
  return agree == checks ? 0 : 2;
}
