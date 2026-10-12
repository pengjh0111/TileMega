// SPDX-License-Identifier: BSD-3-Clause
#pragma once
// 0: mma16, 1: pvswap, 2: pvswap2, 3: simt. The old macro only supports
// archived generated sources; new compiler invocations emit IMPL exclusively.
#ifndef TILEMEGA_ATTENTION_IMPL
#if defined(TILEMEGA_ATTENTION_PVSWAP) && TILEMEGA_ATTENTION_PVSWAP
#define TILEMEGA_ATTENTION_IMPL 1
#else
#define TILEMEGA_ATTENTION_IMPL 0
#endif
#endif
static_assert(TILEMEGA_ATTENTION_IMPL>=0 && TILEMEGA_ATTENTION_IMPL<=3);
