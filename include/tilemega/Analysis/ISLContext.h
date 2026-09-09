// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Presburger domains, §3.5 CuTe->ISL, principle 3.
//
// Hand-built RAII around the isl/barvinok C API.  Deliberately not
// `isl-noexceptions.h`: barvinok is C-API-only, and mixing its raw pointer
// ownership with isl's C++ wrapper's ownership model is a silent
// use-after-free (confirmed during dependency evaluation, see
// docs/DEPENDENCIES.md and docs/experiments/P3_ISL/).
#pragma once

#include <string>

struct isl_ctx;

namespace tilemega::analysis {

/// Owns one isl_ctx.  isl objects built from a context must not outlive it;
/// every wrapper in this file that borrows a context documents that rule
/// rather than hiding it.
class IslContext {
 public:
  IslContext();
  ~IslContext();
  IslContext(IslContext const&) = delete;
  IslContext& operator=(IslContext const&) = delete;
  IslContext(IslContext&&) = delete;
  IslContext& operator=(IslContext&&) = delete;

  isl_ctx* raw() const { return ctx_; }
  int ReferenceCount() const;

 private:
  isl_ctx* ctx_;
  IslContext* previous_ = nullptr;
};

/// Borrows the innermost caller-owned context on this thread. Tools must
/// construct IslContext before their MLIR context and analysis values.
/// Missing ownership is an error, never a static-lifetime allocation.
IslContext& SharedIslContext();

class IslReferenceAudit {
 public:
  explicit IslReferenceAudit(char const* operation);
  ~IslReferenceAudit();
 private:
  IslContext& context_;
  char const* operation_;
  int before_;
};

}  // namespace tilemega::analysis
