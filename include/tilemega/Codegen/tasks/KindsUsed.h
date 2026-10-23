// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/tasks/KindsUsed.h
//
// MX4 prompt §5.2: the generated source states, for every TaskKind, whether
// the plan runs it (TILEMEGA_KIND_USED_<Kind> 0/1, Codegen.cpp
// emitTaskKindRuntime). A kind the plan does not run has no TaskBody alias,
// no TaskSmem member and no dispatch branch in ModelHarness.cuh, so its
// storage cannot size the union. Sources that say nothing (tests, hand-
// written harnesses) keep every kind.
#pragma once

#ifndef TILEMEGA_KIND_USED_Gemm
#define TILEMEGA_KIND_USED_Gemm 1
#endif
#ifndef TILEMEGA_KIND_USED_RMSNorm
#define TILEMEGA_KIND_USED_RMSNorm 1
#endif
#ifndef TILEMEGA_KIND_USED_RoPE
#define TILEMEGA_KIND_USED_RoPE 1
#endif
#ifndef TILEMEGA_KIND_USED_KVAppend
#define TILEMEGA_KIND_USED_KVAppend 1
#endif
#ifndef TILEMEGA_KIND_USED_Elementwise
#define TILEMEGA_KIND_USED_Elementwise 1
#endif
#ifndef TILEMEGA_KIND_USED_Attention
#define TILEMEGA_KIND_USED_Attention 1
#endif
#ifndef TILEMEGA_KIND_USED_GemmCombine
#define TILEMEGA_KIND_USED_GemmCombine 1
#endif
#ifndef TILEMEGA_KIND_USED_GemmAdd
#define TILEMEGA_KIND_USED_GemmAdd 1
#endif
#ifndef TILEMEGA_KIND_USED_GemmRMSNorm
#define TILEMEGA_KIND_USED_GemmRMSNorm 1
#endif
#ifndef TILEMEGA_KIND_USED_RoPEKVAppend
#define TILEMEGA_KIND_USED_RoPEKVAppend 1
#endif
#ifndef TILEMEGA_KIND_USED_Add
#define TILEMEGA_KIND_USED_Add 1
#endif
#ifndef TILEMEGA_KIND_USED_Embedding
#define TILEMEGA_KIND_USED_Embedding 1
#endif
#ifndef TILEMEGA_KIND_USED_QKNorm
#define TILEMEGA_KIND_USED_QKNorm 1
#endif
#ifndef TILEMEGA_KIND_USED_FusedAttention
#define TILEMEGA_KIND_USED_FusedAttention 1
#endif
#ifndef TILEMEGA_KIND_USED_AttentionMerge
#define TILEMEGA_KIND_USED_AttentionMerge 1
#endif
#ifndef TILEMEGA_KIND_USED_ArgmaxReduce
#define TILEMEGA_KIND_USED_ArgmaxReduce 1
#endif
// The DM kinds (kDepthwiseConv ... kLayoutConvert, kDwPwFused) are already
// instantiated per plan by DispatchDmStage (TILEMEGA_DM_STAGE_DISPATCH).

/// `#if TILEMEGA_KIND_USED(Attention)`.
#define TILEMEGA_KIND_USED(kind) TILEMEGA_KIND_USED_##kind
