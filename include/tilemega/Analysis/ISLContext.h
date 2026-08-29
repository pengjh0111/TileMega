// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// isl_ctx 的生命周期、错误处理，以及 isl 对象的 RAII 封装。
//
// 接口选择（骨架 Phase 0 决策）：直接用 isl 的 C API，自建薄 RAII 封装，
// **不用** isl-noexceptions.h 的官方 C++ 绑定。理由：
//   1. barvinok 的基数计数接口（P3.3 的 wait/fan-out 要用）只有 C API。
//      混用两套所有权模型会在边界上出错，而所有权错误在 isl 里表现为
//      静默的 use-after-free 或泄漏，极难定位。
//   2. isl 的 C++ 绑定 API 覆盖面随版本变动，升级 isl 时会破损。
//
// isl 的 C API 所有权约定（必须背熟，封装就是为了不再手工遵守它）：
//   __isl_take  参数所有权转移给被调方，调用后指针失效
//   __isl_keep  被调方只借用，调用方仍持有
//   __isl_give  返回值所有权转移给调用方，调用方负责释放
//
// 用法：
//   ISLContext ctx;
//   auto m = ctx.parseMap("{ [i,j] -> [i] }");
//   isl_map *raw = m.get();          // 借用，不转移
//   isl_map *owned = m.release();    // 转移出去，之后由你负责

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

/// isl 对象的 RAII 持有者。语义等同 unique_ptr：只移动，不复制。
///
/// 之所以不用 unique_ptr 加自定义 deleter，是因为 isl 的释放函数每个类型
/// 各不相同（isl_map_free / isl_set_free / ...），用模板参数传函数指针的
/// 写法在这里更直白，也让 get()/release() 的语义与 isl 的 take/keep 约定
/// 一一对应。
/// 注意 Free 的签名是 T *(*)(T *) 而非 void(*)(T *)：isl 的 *_free 函数
/// 统一返回 __isl_null T*（永远是 nullptr），方便写 p = isl_map_free(p)。
template <typename T, T *(*Free)(T *)> class ISLRef {
public:
  ISLRef() = default;
  /// 接管 obj 的所有权（对应 __isl_take）。
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

  /// 借用底层指针（对应 __isl_keep）。不转移所有权，不要对它调 *_free。
  T *get() const { return obj_; }

  /// 交出所有权（对应把本对象作为 __isl_take 参数传出去）。
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

/// 持有一个 isl_ctx，负责它的生命周期与错误策略。
///
/// 重要：一个进程里所有相互运算的 isl 对象必须来自**同一个** isl_ctx。
/// 跨 ctx 的对象做运算是未定义行为，isl 不会报错。所以 TileMega 里
/// ISLContext 应当按分析会话持有一个，不要随手新建。
class ISLContext {
public:
  ISLContext();
  ~ISLContext();

  ISLContext(const ISLContext &) = delete;
  ISLContext &operator=(const ISLContext &) = delete;

  isl_ctx *get() const { return ctx_; }

  /// 解析 isl 语法的 map，例如 "[S] -> { [i,j] -> [i] : 0 <= i < S }"。
  /// 解析失败返回空 ISLMap，错误信息可用 lastError() 取。
  ISLMap parseMap(const std::string &str);
  ISLSet parseSet(const std::string &str);

  /// 取最近一次 isl 错误的描述；无错误时为空串。
  std::string lastError() const;

  /// 是否有未清理的 isl 错误。
  bool hasError() const;

  /// 清除错误状态。
  void clearError();

private:
  isl_ctx *ctx_ = nullptr;
};

} // namespace tilemega

#endif // TILEMEGA_ANALYSIS_ISLCONTEXT_H
