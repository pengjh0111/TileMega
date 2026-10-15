// SPDX-License-Identifier: BSD-3-Clause
//
// TileMega -- Target/CalibrationMaca.cpp
//
// xcore1000 calibration entry points not yet ported (ported groups live in
// their own .cu files, built by cucc). The NVIDIA microbenchmarks (PTX,
// CUTLASS collectives) are not part of this build; each group raises until it
// is ported onto the MetaX primitive layer, so a target never receives a
// coefficient that was not measured on this device.
#include <tilemega/Target/Calibration.h>

#include <stdexcept>
#include <string>

namespace tilemega::calib {
namespace {
[[noreturn]] void Unported(char const* group) {
  throw std::runtime_error(std::string("calibration group '") + group +
                           "' is not ported to xcore1000");
}
}  // namespace


void MeasureInterference(TargetSpec&, Options const&, std::ostream&) { Unported("interference"); }
void MeasureFP32PartialCombine(TargetSpec&, Options const&, std::ostream&) { Unported("fp32-combine"); }
void MeasureL2Prefetch(TargetSpec&, Options const&, std::ostream&) { Unported("l2-prefetch"); }
void MeasureServingPagedBodies(TargetSpec&, Options const&, std::ostream&) { Unported("serving-paged"); }

}  // namespace tilemega::calib
