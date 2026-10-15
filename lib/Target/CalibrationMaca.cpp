// SPDX-License-Identifier: BSD-3-Clause
//
// TileMega -- Target/CalibrationMaca.cpp
//
// xcore1000 calibration entry points. The NVIDIA microbenchmarks (PTX,
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

std::vector<TargetSpec::StreamKPoint> StreamKShapes(bool) { return {}; }

void MeasurePipelines(TargetSpec&, Options const&, std::ostream&) { Unported("pipelines"); }
void MeasureSync(TargetSpec&, Options const&, std::ostream&) { Unported("sync"); }
void MeasureInterference(TargetSpec&, Options const&, std::ostream&) { Unported("interference"); }
void MeasureStreamK(TargetSpec&, Options const&, std::ostream&) { Unported("streamk"); }
void MeasureFP32PartialCombine(TargetSpec&, Options const&, std::ostream&) { Unported("fp32-combine"); }
void MeasureL2Prefetch(TargetSpec&, Options const&, std::ostream&) { Unported("l2-prefetch"); }
void MeasureInflight(TargetSpec&, Options const&, std::ostream&) { Unported("inflight"); }
void MeasureServingTaskBodies(TargetSpec&, Options const&, std::ostream&) { Unported("serving-task"); }
void MeasureServingPagedBodies(TargetSpec&, Options const&, std::ostream&) { Unported("serving-paged"); }
void MeasureServingWaitPolicy(TargetSpec&, Options const&, std::ostream&) { Unported("serving-wait"); }
void MeasureServingHop(TargetSpec&, Options const&, std::ostream&) { Unported("serving-hop"); }
void MeasureServingEvents(TargetSpec&, Options const&, std::ostream&) { Unported("serving-events"); }
void Run(TargetSpec&, Options const&, std::ostream&) { Unported("all"); }

}  // namespace tilemega::calib
