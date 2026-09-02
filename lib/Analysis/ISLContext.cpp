// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>

#include <isl/ctx.h>

namespace tilemega::analysis {

IslContext::IslContext() : ctx_(isl_ctx_alloc()) {}
IslContext::~IslContext() { isl_ctx_free(ctx_); }

IslContext& SharedIslContext() {
  static IslContext context;
  return context;
}

}  // namespace tilemega::analysis
