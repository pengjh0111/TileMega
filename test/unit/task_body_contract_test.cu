// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/AttentionChunkTaskBody.h>
#include <tilemega/Codegen/tasks/ElementwiseTaskBody.h>
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Codegen/tasks/GemmTaskBody.h>
#include <tilemega/Codegen/tasks/KVAppendTaskBody.h>
#include <tilemega/Codegen/tasks/RMSNormTaskBody.h>
#include <tilemega/Codegen/tasks/RoPETaskBody.h>

#include <algorithm>

using Arch = tilemega::arch::Sm89;
using Gemm = tilemega::codegen::GemmTaskBody<Arch, 3>;
union AllTaskStorage;
using RuntimeGemm = tilemega::codegen::GemmStageTaskBody<Arch, AllTaskStorage, 256>;
using Norm = tilemega::codegen::RMSNormTaskBody<Arch, AllTaskStorage, 256>;
using Rope = tilemega::codegen::RoPETaskBody<Arch, AllTaskStorage, 256>;
using KV = tilemega::codegen::KVAppendTaskBody<Arch, AllTaskStorage, 256>;
using Elem = tilemega::codegen::ElementwiseTaskBody<Arch, AllTaskStorage, 256>;
using Attn = tilemega::codegen::AttentionTaskBody<Arch, AllTaskStorage, 256>;

union AllTaskStorage {
  Gemm::SharedStorage gemm;
  RuntimeGemm::SharedStorage runtime_gemm;
  Norm::SharedStorage norm;
  Rope::SharedStorage rope;
  KV::SharedStorage kv;
  Elem::SharedStorage elementwise;
  Attn::SharedStorage attention;
};

constexpr std::size_t kMaxStorage = std::max({
    sizeof(Gemm::SharedStorage), sizeof(RuntimeGemm::SharedStorage),
    sizeof(Norm::SharedStorage),
    sizeof(Rope::SharedStorage), sizeof(KV::SharedStorage),
    sizeof(Elem::SharedStorage), sizeof(Attn::SharedStorage)});
static_assert(sizeof(AllTaskStorage) == kMaxStorage);
static_assert(Gemm::kLegal && RuntimeGemm::kLegal && Norm::kLegal &&
              Rope::kLegal && KV::kLegal &&
              Elem::kLegal && Attn::kLegal);
static_assert(Gemm::kSmemBytes == sizeof(Gemm::SharedStorage));
static_assert(Gemm::kNumThreads > 0);

int main() { return 0; }
