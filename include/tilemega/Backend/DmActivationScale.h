// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Backend/DmMoeOperands.h>
#include <tilemega/Target/Maca/Bf16.h>
#include <type_traits>
#include <utility>

namespace tilemega::backend {
struct DmUnscaledActivation {
  template<class Fragment,class Coordinates>
  __device__ void operator()(Fragment,Coordinates const&,int) const {}
};
template<class T,class=void>struct HasDmActivationScale:std::false_type {};
template<class T>struct HasDmActivationScale<T,std::void_t<decltype(std::declval<T>().a_scale)>>:std::true_type {};

// SCA's multiply rounds to BF16 before the next GEMM consumes it. On
// xcore1000 the scale is applied to each loaded A vector inside the DM
// operand loaders (Backend/ServingDmGemm.h DmActivationScaler); there is no ldmatrix
// fragment transform.
} // namespace tilemega::backend
