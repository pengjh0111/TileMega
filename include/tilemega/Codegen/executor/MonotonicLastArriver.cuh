// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/EpochLastArriver.cuh>
namespace tilemega::codegen::executor {
// Run is the contribution=1 specialization of the weighted epoch protocol.
using MonotonicLastArriver=EpochLastArriver;
}
