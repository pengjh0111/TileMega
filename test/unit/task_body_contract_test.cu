// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/AttentionChunkTaskBody.h>
#include <tilemega/Codegen/tasks/ElementwiseTaskBody.h>
#include <tilemega/Codegen/tasks/GemmTaskBody.h>
#include <tilemega/Codegen/tasks/KVAppendTaskBody.h>
#include <tilemega/Codegen/tasks/RMSNormTaskBody.h>
#include <tilemega/Codegen/tasks/RoPETaskBody.h>

#include <algorithm>

using Arch = tilemega::arch::Sm89;
using Gemm = tilemega::codegen::GemmTaskBody<Arch, 3>;
using Norm = tilemega::codegen::RMSNormTaskBody<Arch>;
using Rope = tilemega::codegen::RoPETaskBody<Arch>;
using KV = tilemega::codegen::KVAppendTaskBody<Arch>;
using Elem = tilemega::codegen::ElementwiseTaskBody<Arch>;
using Attn = tilemega::codegen::AttentionChunkTaskBody<Arch, 2>;

union AllTaskStorage {
  Gemm::SharedStorage gemm;
  Norm::SharedStorage norm;
  Rope::SharedStorage rope;
  KV::SharedStorage kv;
  Elem::SharedStorage elementwise;
  Attn::SharedStorage attention;
};

constexpr std::size_t kMaxStorage = std::max({
    sizeof(Gemm::SharedStorage), sizeof(Norm::SharedStorage),
    sizeof(Rope::SharedStorage), sizeof(KV::SharedStorage),
    sizeof(Elem::SharedStorage), sizeof(Attn::SharedStorage)});
static_assert(sizeof(AllTaskStorage) == kMaxStorage);
static_assert(Gemm::kLegal && Norm::kLegal && Rope::kLegal && KV::kLegal &&
              Elem::kLegal && Attn::kLegal);
static_assert(Gemm::kSmemBytes == sizeof(Gemm::SharedStorage));
static_assert(Gemm::kNumThreads > 0);

int main() { return 0; }
