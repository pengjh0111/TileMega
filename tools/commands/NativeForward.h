// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <string>
namespace tilemega::commands {
// Native DNN forward (prompt §10 N-1): one typed Sm90ForwardProgram per
// model and batch, either from fixed coordinates or from the forward solver's
// rule-pruned search with a measured top-4 selection.
int CompileNativeForwardFixed(std::string const& input,std::string const& output,
    std::string const& coordinates,std::string const& target,int batch,bool fuse,
    std::string const& dump_cg,int split_compile);
struct NativeForwardSolveOptions {
  std::string input,output,target,dump_cg,measure_command,solve_command;
  int batch=1,split_compile=2,measure_top=4;
  bool measured=true,fuse=true;
};
int SolveNativeForward(NativeForwardSolveOptions const&);
}
