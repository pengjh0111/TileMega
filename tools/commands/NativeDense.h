// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <stdexcept>
#include <string>
namespace tilemega::commands {
int CompileNativeDenseFixed(std::string const& input,std::string const& output,
    std::string const& coordinates,std::string const& target,int batch,
    std::string const& dump_cg,int split_compile);
bool IsNativeDenseInput(std::string const& input);
struct NativeDenseSolveOptions {
  std::string input,output,target,dump_cg,measure_command,solve_command;
  int batch=1,split_compile=2,measure_top=4;
  // Megatron tensor parallel degree; must match the export manifest's.
  int tp=1;
  bool measured=true;
};
int SolveNativeDense(NativeDenseSolveOptions const&);
}
