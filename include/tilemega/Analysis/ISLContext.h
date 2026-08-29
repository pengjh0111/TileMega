// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// isl_ctx lifetime, error handling, and RAII wrappers for isl objects.
//
// Interface choice (Phase 0 decision): use isl's C API directly with a thin
// RAII wrapper, *not* the official C++ bindings in isl-noexceptions.h. Reasons:
//   1. barvinok's cardinality-counting interface (needed by P3.3 for wait /
//      fan-out) is C-only. Mixing two ownership models invites errors at the
//      boundary, and ownership errors in isl surface as silent use-after-free
//      or leaks, which are very hard to track down.
//   2. The API surface of isl's C++ bindings shifts between releases, so it
//      breaks on isl upgrades.
//
// isl's C ownership conventions (memorise these; the wrapper exists so you no
// longer have to honour them by hand):
//   __isl_take  ownership transfers to the callee; the pointer is dead after
//   __isl_keep  the callee only borrows; the caller still owns it
//   __isl_give  ownership transfers to the caller, who must free it
//
// Usage:
//   ISLContext ctx;
//   auto m = ctx.parseMap("{ [i,j] -> [i] }");
//   isl_map *raw = m.get();          // borrow, no transfer
//   isl_map *owned = m.release();    // transfer out; you own it now

#ifndef TILEMEGA_ANALYSIS_ISLCONTEXT_H
#define TILEMEGA_ANALYSIS_ISLCONTEXT_H

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>

#include <memory>
#include <string>
#include <utility>

namespace tilemega {

/// RAII holder for an isl object. Move-only, like unique_ptr.
///
/// This is not unique_ptr with a custom deleter because isl uses a distinct
/// free function per type (isl_map_free / isl_set_free / ...). Passing the
/// function as a template parameter is more direct here, and it keeps
/// get()/release() in one-to-one correspondence with isl's keep/take
/// conventions.
///
/// Note that Free is typed T *(*)(T *) rather than void(*)(T *): isl's *_free
/// functions all return __isl_null T* (always nullptr) so that `p =
/// isl_map_free(p)` reads well.
template <typename T, T *(*Free)(T *)> class ISLRef {
public:
  ISLRef() = default;
  /// Takes ownership of obj (corresponds to __isl_take).
  explicit ISLRef(T *obj) : obj_(obj) {}

  ISLRef(const ISLRef &) = delete;
  ISLRef &operator=(const ISLRef &) = delete;

  ISLRef(ISLRef &&other) noexcept : obj_(other.obj_) { other.obj_ = nullptr; }
  ISLRef &operator=(ISLRef &&other) noexcept {
    if (this != &other) {
      reset();
      obj_ = other.obj_;
      other.obj_ = nullptr;
    }
    return *this;
  }

  ~ISLRef() { reset(); }

  /// Borrows the underlying pointer (corresponds to __isl_keep). Ownership is
  /// not transferred; do not call *_free on the result.
  T *get() const { return obj_; }

  /// Gives up ownership (corresponds to passing this object as an __isl_take
  /// argument).
  [[nodiscard]] T *release() {
    T *tmp = obj_;
    obj_ = nullptr;
    return tmp;
  }

  void reset(T *obj = nullptr) {
    if (obj_)
      (void)Free(obj_);
    obj_ = obj;
  }

  explicit operator bool() const { return obj_ != nullptr; }

private:
  T *obj_ = nullptr;
};

using ISLMap = ISLRef<isl_map, isl_map_free>;
using ISLSet = ISLRef<isl_set, isl_set_free>;
using ISLUnionMap = ISLRef<isl_union_map, isl_union_map_free>;
using ISLSpace = ISLRef<isl_space, isl_space_free>;

/// Owns an isl_ctx and defines its lifetime and error policy.
///
/// Important: within a process, all isl objects that interact must come from
/// the *same* isl_ctx. Operating across contexts is undefined behaviour and
/// isl will not diagnose it. So TileMega should hold one ISLContext per
/// analysis session rather than creating them ad hoc.
class ISLContext {
public:
  ISLContext();
  ~ISLContext();

  ISLContext(const ISLContext &) = delete;
  ISLContext &operator=(const ISLContext &) = delete;

  isl_ctx *get() const { return ctx_; }

  /// Parses a map in isl syntax, e.g. "[S] -> { [i,j] -> [i] : 0 <= i < S }".
  /// Returns an empty ISLMap on failure; see lastError() for the reason.
  ISLMap parseMap(const std::string &str);
  ISLSet parseSet(const std::string &str);

  /// Description of the most recent isl error, or an empty string if none.
  std::string lastError() const;

  /// Whether an isl error is pending.
  bool hasError() const;

  /// Clears the pending error state.
  void clearError();

private:
  isl_ctx *ctx_ = nullptr;
};

} // namespace tilemega

#endif // TILEMEGA_ANALYSIS_ISLCONTEXT_H
