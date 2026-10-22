// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/MxDecodeProgram.h
//
// The MX-4 static decode program generator (MX4 prompt §3.9): reads a dense
// decode module's `tilemega.model_plan` and the per-class configuration the
// solver (or a pin) wrote, and emits the plan source -- the constexpr plan
// type of Codegen/tasks/mx/DecodeProgram.cuh, the external buffer table and
// the runtime ABI (DecodeRuntime.cuh). Nothing else is read.
#pragma once

#include <mlir/IR/BuiltinOps.h>

#include <string>
#include <vector>

namespace tilemega::codegen {

/// One GEMM class's coordinates (§4.1).
struct MxDecodeClass {
  int tn = 0, d = 0;
  bool stream_k = false;
};

/// The solver's choice for a plan (§4.4): every GEMM class and Tmin.
struct MxDecodeConfig {
  MxDecodeClass qkv, o, gate_up, down, head;
  int tmin = 64;
};

/// The dense decoder the plan describes, checked against the program's
/// fixed stage sequence (§3.9).
struct MxDecodeModel {
  int layers = 0, hidden = 0, heads = 0, hkv = 0, head_dim = 0, q_per_kv = 0, inter = 0, vocab = 0;
  int capacity = 0;
  bool qk_norm = false;
  double epsilon = 0;
  // Indices into model_plan.buffers.
  int tokens = -1, rope_cos = -1, rope_sin = -1, embed = -1, head = -1;
  std::vector<int> qkv, o, gate_up, down, q_norm, k_norm, k_cache, v_cache;
};

/// Throws std::invalid_argument when the module is not the dense decoder
/// [Embedding, (Gemm qkv, FusedAttention, [AttentionMerge], Gemm o,
/// Gemm gate_up, Gemm down) x L, Gemm head, ArgmaxReduce] with norm_ss on
/// qkv, gate_up and head (gamma folded into the weights).
MxDecodeModel AnalyzeMxDecode(mlir::ModuleOp module);

/// Writes the choice: module `tmexec.executor = "static_stage"`,
/// `tmexec.attention_tmin`, and `tmexec.implementation = {tn, d, part}` on
/// every GEMM tile space.
void WriteMxDecodeConfig(mlir::ModuleOp module, MxDecodeConfig const& config);
MxDecodeConfig ReadMxDecodeConfig(mlir::ModuleOp module);

/// Throws std::invalid_argument for a coordinate outside §3.5 / §4.1.
void CheckMxDecodeConfig(MxDecodeModel const& model, int batch, MxDecodeConfig const& config, int grid);

/// The plan source (plan.so.cu) of one batch size.
std::string EmitMxDecodeProgram(mlir::ModuleOp module, int batch, int past_lo, int past_hi, int grid);

}  // namespace tilemega::codegen
