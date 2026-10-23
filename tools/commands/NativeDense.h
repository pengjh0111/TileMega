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
  std::string input,output,target,dump_cg,measure_command,evaluate_configs,solve_command;
  int batch=1,split_compile=2,measure_top=6,passes=12,budget_ms=600000;
  bool measured=true;
};
inline void ValidateNativeSelectionSearch(bool measured,bool converged,bool budget_exhausted) {
  if(measured && (!converged || budget_exhausted))
    throw std::invalid_argument("measured native selection requires a converged search without "
        "budget exhaustion; retain the search receipt and increase --search-passes or --search-budget-ms");
}
int SolveNativeDense(NativeDenseSolveOptions const&);
}
