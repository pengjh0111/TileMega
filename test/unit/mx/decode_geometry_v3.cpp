// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeGeometryV2.h>
using namespace tilemega::backend::mx;
static_assert(!MxDepthFits(128,4));
static_assert(MxDepthFits(128,4,192));
static_assert(!MxDepthFits(128,5,192));
static_assert(!MxGemmLegalV2(1024,2048,1,128,8,1,MxOpV2::kSwiGLU,MxPart::kTileRange));
static_assert(MxGemmLegalV3(1024,2048,1,128,8,4,MxOpV2::kSwiGLU,MxPart::kTileRange,true));
static_assert(!MxGemmLegalV3(1024,2048,1,128,8,4,MxOpV2::kSwiGLU,MxPart::kTileRange,false));
static_assert(!MxGemmLegalV2(151936,2048,16,128,8,4,MxOpV2::kStore,MxPart::kTileRange));
static_assert(MxGemmLegalV3(151936,2048,16,128,8,4,MxOpV2::kStore,MxPart::kTileRange,false));
static_assert(!MxGemmLegalV3(1024,2048,1,128,8,4,MxOpV2::kStore,MxPart::kStreamK,true));
int main(){}
