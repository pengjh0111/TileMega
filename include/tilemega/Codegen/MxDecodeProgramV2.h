// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/MxDecodeProgram.h>

namespace tilemega::codegen {

struct MxDecodeClassV2 {
  int tn = 0, wn = 1, d = 1;
  bool stream_k = false;
  // -1 derives Pw from measured r_next; explicit values support resource retries.
  int prefetch = -1, r_next = -1;
};

struct MxDecodeConfigV2 {
  MxDecodeClassV2 qkv, o, gate_up, down, head;
  int b_min = 1, attn_prefetch = 0, threads = 576;
  int attn_version = 2, split = 1;
};

/// Validate production geometry, resource-derived payloads and wait counts.
/// Explicit Pw permits resource probes; the final-selection caller must also
/// validate its qualified measured R_next and post-build registers/stack.
void CheckMxDecodeConfigV2(MxDecodeModel const& model, int batch, MxDecodeConfigV2 const& config, int grid);
/// Resolve a resource-derived Pw without inventing an unmeasured R_next.
int MxDecodePrefetchV2(MxDecodeClassV2 const& config, int n, int k, int grid, int threads);
int MxDecodeBodySmemBytesV2(MxDecodeModel const& model, int batch, MxDecodeConfigV2 const& config);
std::vector<MxDecodeBuffer> MxDecodeBuffersV2(mlir::ModuleOp module, MxDecodeConfigV2 const& config);
std::string EmitMxDecodeProgramV2(mlir::ModuleOp module, int batch, int past_lo, int past_hi, int grid,
                                MxDecodeConfigV2 const& config);

}  // namespace tilemega::codegen
