// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Target/Maca/DevicePrimitives.h>

namespace tilemega::codegen::executor {

template <class Sync>
struct LastArriverV2 {
  // shared_last is independent of the reduction's work area. The following
  // grid barrier separates the ticket reset from the ticket's next use.
  template <class Reduction>
  __device__ __forceinline__ static bool Run(unsigned* ticket, unsigned producers, unsigned* shared_last,
                                             Reduction reduce) {
    Sync::CtaSync();
    if (Sync::IsRole()) {
      unsigned const previous = Sync::ReleaseAdd(ticket, 1u);
      if (!producers || previous >= producers) maca::Trap();
      *shared_last = previous + 1 == producers;
    }
    Sync::CtaSync();
    bool const last = *shared_last != 0;
    if (last) {
      if (Sync::IsRole()) Sync::Acquire();
      Sync::CtaSync();
      reduce();
      if (Sync::IsRole()) atomicExch(ticket, 0u);
    }
    return last;
  }
};

}  // namespace tilemega::codegen::executor
