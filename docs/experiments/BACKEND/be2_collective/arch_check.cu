// SPDX-License-Identifier: BSD-3-Clause
// R8 BE-2 / A-f: instantiate the GEMM collective each architecture selects,
// through the repository's own headers, and report what came out. Compiled
// once per -arch; the check itself runs on the CPU, so an architecture with
// no device here is still covered.
#include <cstdio>
#include <tilemega/Backend/CutlassGemmCandidate.h>
#include <tilemega/Target/ArchDispatch.h>

using namespace tilemega;

template <class Arch>
static void Report(char const* tag) {
  using Candidate = backend::TypedGemmCandidate<true, 64, 128, 64, 3, Arch>;
  using Collective = typename Candidate::Collective;
  std::printf("ARCH_COLLECTIVE arch=%s builder=%d path=\"%s\" legal=%d "
              "threads=%d mainloop_smem=%d priced=%d\n",
              tag, static_cast<int>(arch::Caps<Arch>::kBf16CollectiveBuilder),
              Collective::kPath, static_cast<int>(Candidate::kShapeLegal),
              Candidate::kThreads, Collective::kSmemBytes,
              static_cast<int>(Candidate::kPricedCollective));
}

int main() {
#if TILEMEGA_CHECK_ARCH == 800
  Report<arch::Sm80>("sm_80");
#elif TILEMEGA_CHECK_ARCH == 890
  Report<arch::Sm89>("sm_89");
#elif TILEMEGA_CHECK_ARCH == 900
  Report<arch::Sm90>("sm_90");
#elif TILEMEGA_CHECK_ARCH == 1000
  Report<arch::Sm100>("sm_100");
#elif TILEMEGA_CHECK_ARCH == 1200
  Report<arch::Sm120>("sm_120");
#else
#  error "TILEMEGA_CHECK_ARCH must name one of the supported architectures"
#endif
  return 0;
}
