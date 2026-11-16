// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <stdexcept>
#include <string>
namespace tilemega::commands {
int CompileNativeDenseFixed(std::string const& input,std::string const& output,
    std::string const& coordinates,std::string const& target,int batch,
    std::string const& dump_cg,int split_compile,bool kernel_sequence=false);
int CompileNativeDenseProbe(std::string const& input,std::string const& output,
    std::string const& coordinates,std::string const& target,int batch,
    std::string const& family,std::string const& entry_limit,int split_compile);
bool IsNativeDenseInput(std::string const& input);
struct NativeDenseSolveOptions {
  std::string input,output,target,dump_cg,measure_command,solve_command;
  int batch=1,split_compile=2,measure_top=4;
  // Megatron tensor parallel degree; must match the export manifest's.
  int tp=1;
  bool measured=true;
  // P3 and the ablation arms (continuation 1 sections 3 and 5, appendix C).
  // `pipeline` is the default for native decode; `additive` reproduces the
  // R3-R5 selections and is never used for a final choice this round.
  std::string search="pipeline";
  std::string analysis="reuse";
  // AB-2: rank by the sum of the operators' own times (Eq. 7 per operator)
  // instead of the program's completion (Eq. 8).
  bool operator_sum=false;
  // AB-4a: forbid Table 1's structural rows other than Retile.
  bool retile_only=false;
  // AB-4b: realize every coupling as an aggregate (V3) boundary.
  bool realize_aggregate=false;
  // AB-1: pin each family's geometry to a file of locally chosen theta.
  std::string pin_theta;
  std::string seed_coordinates;
  double search_budget_ms=600000;
};
int SolveNativeDense(NativeDenseSolveOptions const&);
}
