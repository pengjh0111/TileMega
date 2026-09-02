// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2.7 the Llama decoder layer coupling table.
#pragma once

#include <string>

#include <tilemega/Analysis/ClosedForm.h>
#include <tilemega/Analysis/TensorSpace.h>

namespace tilemega::analysis {

/// The symbolic shape and granularity of a decoder layer.  Everything is a
/// ClosedForm so the derived quantities stay parameterized in (theta, g);
/// nothing here is a substituted scalar (F-14, invariant I1).
struct DecoderShape {
  ClosedForm S = ClosedForm::Symbol("S");          ///< tokens in this pass
  ClosedForm H = ClosedForm::Symbol("H");          ///< hidden size
  ClosedForm n_h = ClosedForm::Symbol("n_h");      ///< query heads
  ClosedForm n_kv = ClosedForm::Symbol("n_kv");    ///< key/value heads
  ClosedForm group = ClosedForm::Symbol("G");      ///< n_h / n_kv
  ClosedForm d = ClosedForm::Symbol("d");          ///< head dim
  ClosedForm I = ClosedForm::Symbol("I");          ///< MLP intermediate
  ClosedForm L_s = ClosedForm::Symbol("L_s");      ///< KV length after append
  ClosedForm past = ClosedForm::Symbol("past");    ///< KV rows already present
  ClosedForm Tm = ClosedForm::Symbol("Tm");
  ClosedForm Tn = ClosedForm::Symbol("Tn");
  ClosedForm Tkv = ClosedForm::Symbol("Tkv");

  /// The §2.7 instantiation: H=4096, n_h=32, n_kv=8, d=128, I=14336,
  /// Tm=Tn=Tkv=128.  S and L_s stay free.
  static ParamBinding Table27Theta();
  static ParamBinding Table27G();
};

/// One Llama decoder layer at the granularity §2.7 talks about: one node per
/// tensor operation, not one node per FX call_function.  `prefix` namespaces
/// the node names so several layers can be stacked.
OperatorGraph LlamaDecoderLayer(DecoderShape const& shape,
                                std::string const& prefix = "");

/// `layers` stacked decoder layers, wired residual-to-residual.
OperatorGraph LlamaStack(DecoderShape const& shape, int layers);

/// A structurally different model: a plain MLP stack with no attention, no KV
/// cache and no RoPE.  Used to show the generator is driven by the CG rather
/// than by a hardcoded Llama stage sequence.
OperatorGraph MlpStack(DecoderShape const& shape, int blocks);

/// Multi-head attention without GQA (n_kv == n_h) and a two-matrix MLP.
OperatorGraph MhaModel(DecoderShape const& shape, int layers);

/// An artificial Tier 3 case: a gather whose index tensor is only known at run
/// time.  It must degrade to an operator-level barrier, not to a fake affine
/// relation.
/// `data_dependent = false` builds the same graph with an affine index, so a
/// test can compare the relaxed relation against the exact one it must contain.
OperatorGraph GatherModel(DecoderShape const& shape,
                          bool data_dependent = true);

}  // namespace tilemega::analysis
