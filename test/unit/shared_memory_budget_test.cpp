// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/executor/SharedMemoryBudget.h>
#include <limits>

using tilemega::codegen::executor::SharedMemoryRequiresOptIn;
static_assert(!SharedMemoryRequiresOptIn(49152, 0, 49152));
static_assert(SharedMemoryRequiresOptIn(49152, 16, 49152));
static_assert(!SharedMemoryRequiresOptIn(49136, 16, 49152));
static_assert(SharedMemoryRequiresOptIn(49137, 16, 49152));
static_assert(!SharedMemoryRequiresOptIn(98304, 0, 98304));
static_assert(SharedMemoryRequiresOptIn(98304, 16, 98304));
static_assert(SharedMemoryRequiresOptIn(0, 49153, 49152));
static_assert(SharedMemoryRequiresOptIn(std::numeric_limits<std::size_t>::max(), 16, 49152));
int main() { return 0; }
