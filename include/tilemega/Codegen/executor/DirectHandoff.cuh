// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
namespace tilemega::codegen::executor {
// A handoff occupies one page of the current CTA.  The solved schedule must
// place the producer and consumer next to one another on this worker; no
// global event or intermediate global buffer is needed for their shared edge.
template<class Producer,class Consumer>
__device__ void DirectHandoff(void* page,Producer produce,Consumer consume) {
  // A paged decode CTA has an extra loader warp. Only its four compute
  // warps may participate in this named barrier or touch the handoff page.
  if(!IsCompute())return;
  produce(page);
  ComputeSync();
  consume(page);
  ComputeSync();
}
} // namespace tilemega::codegen::executor
