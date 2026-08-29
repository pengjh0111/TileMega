// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tilemega/Analysis/ISLContext.h"

#include <isl/options.h>

namespace tilemega {

ISLContext::ISLContext() : ctx_(isl_ctx_alloc()) {
  // ISL_ON_ERROR_CONTINUE: on error, neither abort nor print. Record the error
  // on the context and let callers query it via hasError() / lastError().
  //
  // Not ISL_ON_ERROR_WARN: TileMega constructs and tests large numbers of
  // possibly-illegal candidate relations (the legality pruning in P4.2 works
  // precisely by trying them). Failure is normal control flow and should not
  // spew warnings to stderr.
  //
  // Not ISL_ON_ERROR_ABORT: a malformed layout string must not kill the
  // compiler process.
  isl_options_set_on_error(ctx_, ISL_ON_ERROR_CONTINUE);
}

ISLContext::~ISLContext() {
  if (ctx_) {
    // isl_ctx_free aborts if objects are still alive (isl's built-in leak
    // check). That is desirable here: it turns "an ISLRef was never destroyed"
    // and similar ownership bugs into an explicit crash rather than a silent
    // leak.
    isl_ctx_free(ctx_);
    ctx_ = nullptr;
  }
}

ISLMap ISLContext::parseMap(const std::string &str) {
  clearError();
  return ISLMap(isl_map_read_from_str(ctx_, str.c_str()));
}

ISLSet ISLContext::parseSet(const std::string &str) {
  clearError();
  return ISLSet(isl_set_read_from_str(ctx_, str.c_str()));
}

bool ISLContext::hasError() const {
  return isl_ctx_last_error(ctx_) != isl_error_none;
}

std::string ISLContext::lastError() const {
  if (!hasError())
    return {};
  const char *msg = isl_ctx_last_error_msg(ctx_);
  return msg ? std::string(msg) : std::string("未知 isl 错误");
}

void ISLContext::clearError() { isl_ctx_reset_error(ctx_); }

} // namespace tilemega
