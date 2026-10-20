// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <string>
namespace tilemega::commands {
int CompileNativeDenseFixed(std::string const& input,std::string const& output,
    std::string const& coordinates,std::string const& target,int batch,
    std::string const& dump_cg,int split_compile);
}
