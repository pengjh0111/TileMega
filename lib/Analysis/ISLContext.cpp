// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tilemega/Analysis/ISLContext.h"

#include <isl/options.h>

namespace tilemega {

ISLContext::ISLContext() : ctx_(isl_ctx_alloc()) {
  // ISL_ON_ERROR_CONTINUE：出错时不 abort、不打印，只把错误记在 ctx 上，
  // 由调用方通过 hasError()/lastError() 查询。
  //
  // 为什么不用默认的 ISL_ON_ERROR_WARN：TileMega 会大量构造并测试
  // 「可能不合法」的候选关系（P4.2 的合法性剪枝就是靠试出来的），
  // 失败是正常控制流，不该往 stderr 刷警告。
  //
  // 为什么不用 ISL_ON_ERROR_ABORT：一个畸形的 layout 字符串不应该
  // 让整个编译器进程死掉。
  isl_options_set_on_error(ctx_, ISL_ON_ERROR_CONTINUE);
}

ISLContext::~ISLContext() {
  if (ctx_) {
    // isl_ctx_free 在仍有对象存活时会 abort（isl 自带的泄漏检查）。
    // 这实际上是件好事：它把「ISLRef 忘了析构」这类所有权 bug
    // 变成一个明确的崩溃，而不是静默泄漏。
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
