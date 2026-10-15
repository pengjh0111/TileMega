// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Calibration.h>
#include <iosfwd>
namespace tilemega::calib {
// Diagnostic fragment only: does not replace cost-model coefficients.
void MeasureHopperDiagnostics(TargetSpec const&, Options const&,
                              std::ostream& raw, std::ostream& fragment);
}
