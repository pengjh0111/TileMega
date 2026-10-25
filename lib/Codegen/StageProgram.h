// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tilemega::codegen {

enum class StageProgramKind { kDecode, kPrompt, kForward };

// Expressions are C++ source in the scope of RunStep<Self>: b, past, smem,
// tokens and the loop index. Prefetch expressions must have no stage producer.
struct StageProgramStage {
  std::string name, body_type, shape_type, operands, prefetch_operands;
  std::string next, next_prefetch_operands;
  int diagnostic_class = 0;
  // The final loop iteration can target a different compile-time body.
  std::string terminal_body, terminal_next, terminal_prefetch_operands;
};
struct StageProgramLoop {
  std::string index = "l", begin = "0", end;
};
struct StageProgramBuffer {
  std::string name, dtype_expr;
  std::uint64_t constant = 0, per_batch = 0;
  std::string recipe, slot_expr;
  int layer = 0;
};
struct StageProgramSpec {
  StageProgramKind kind = StageProgramKind::kDecode;
  std::string namespace_name = "tm_mx_plan", plan_name = "Plan";
  std::vector<std::string> includes;
  std::string binding_declaration, plan_declarations;
  StageProgramLoop loop;
  std::vector<StageProgramStage> stages, tail_stages;
  std::vector<StageProgramBuffer> external_buffers;
  std::string buffer_row_type = "mx::MxBufferRowV2";
  std::string operand_parameters = "Binding const& b, int past, backend::mx::MxSharedTokensV2* tokens";
  std::string run_parameters = "Binding const& b, int past, char* smem, backend::mx::MxSharedTokensV2* tokens";
  std::string entry_prefetch_operands;
  std::string runtime_header;
};

// Emit concrete type aliases and straight-line Run calls inside one layer
// loop. No stage descriptor or virtual stage survives into the device source.
std::string EmitStageProgram(StageProgramSpec const& spec);

}  // namespace tilemega::codegen
