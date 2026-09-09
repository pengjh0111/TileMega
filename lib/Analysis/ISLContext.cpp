// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>

#include <isl/ctx.h>
#include <isl_ctx_private.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace tilemega::analysis {

namespace { thread_local IslContext* current = nullptr; }

IslContext::IslContext() : ctx_(isl_ctx_alloc()), previous_(current) {
  if (!ctx_) throw std::bad_alloc();
  current = this;
}
IslContext::~IslContext() {
  if (current != this || ReferenceCount() != 0) {
    std::fprintf(stderr, "ISL_CONTEXT invalid_lifetime remaining=%d\n", ReferenceCount());
    std::abort();
  }
  if (std::getenv("TILEMEGA_ISL_AUDIT"))
    std::fprintf(stderr, "ISL_CONTEXT remaining=0\n");
  current = previous_;
  isl_ctx_free(ctx_);
}

// The stack is built from the pinned bundled isl; this diagnostic uses its
// matching private header, never a guessed object layout or copied ABI.
int IslContext::ReferenceCount() const { return ctx_->ref; }

IslContext& SharedIslContext() {
  if (!current) throw std::logic_error("analysis requires a caller-owned IslContext");
  return *current;
}

IslReferenceAudit::IslReferenceAudit(char const* operation)
    : context_(SharedIslContext()), operation_(operation),
      before_(context_.ReferenceCount()) {}

IslReferenceAudit::~IslReferenceAudit() {
  int const after = context_.ReferenceCount();
  if (after != before_) {
    std::fprintf(stderr, "ISL_REFERENCE_LEAK function=%s before=%d after=%d\n",
                 operation_, before_, after);
    std::abort();
  }
}

}  // namespace tilemega::analysis
