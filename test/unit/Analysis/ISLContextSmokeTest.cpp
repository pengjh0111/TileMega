// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ISL 桥接层的冒烟测试。验证三件在 Phase 3 会被反复依赖的事：
//   1. RAII 封装的所有权正确（ctx 析构时 isl 的泄漏检查不 abort）
//   2. 错误策略是 CONTINUE 而不是 abort —— 畸形输入不能杀进程
//   3. barvinok 的参数化基数计数可用 —— 这是 P3.3 算 wait/fan-out 的基础

#include "tilemega/Analysis/ISLContext.h"

#include <barvinok/isl.h>
#include <isl/aff.h>
#include <isl/polynomial.h>
#include <isl/val.h>

#include <cstdio>
#include <cstring>
#include <string>

using namespace tilemega;

static int failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("  [失败] %s\n", msg);                                       \
      ++failures;                                                              \
    } else {                                                                   \
      std::printf("  [通过] %s\n", msg);                                       \
    }                                                                          \
  } while (0)

int main() {
  ISLContext ctx;

  // --- 1. 基本解析与移动语义 ------------------------------------------------
  std::printf("1. 解析与 RAII\n");
  {
    ISLMap m = ctx.parseMap("{ [i,j] -> [i] : 0 <= i < 128 and 0 <= j < 64 }");
    CHECK(bool(m), "解析仿射 map");

    ISLMap moved = std::move(m);
    CHECK(bool(moved) && !m, "移动后源置空、目标持有");
  } // moved 在此析构；若所有权有误，ctx 析构时 isl 会 abort

  // --- 2. 错误策略：畸形输入不得 abort --------------------------------------
  std::printf("2. 错误处理\n");
  {
    ISLMap bad = ctx.parseMap("这不是合法的 isl 语法 {{{");
    CHECK(!bad, "畸形输入返回空而不是崩溃");
    CHECK(ctx.hasError(), "错误状态被记录");
    ctx.clearError();
    CHECK(!ctx.hasError(), "错误可清除");
  }

  // --- 3. Dep = P^-1 ∘ A 的最小演练 ----------------------------------------
  //     骨架 §3.6 边 1（RMSNorm1 → Wq）：消费者 task (m,n) 依赖生产者 task m。
  std::printf("3. 依赖推导 Dep = P^-1 . A\n");
  {
    // A_cons: 消费者 task 坐标 -> 它读的元素行块
    ISLMap A = ctx.parseMap("{ [m,n] -> [m] : 0 <= m < 8 and 0 <= n < 4 }");
    // P_prod: 生产者 task 坐标 -> 它写的元素行块（此处恒等）
    ISLMap P = ctx.parseMap("{ [p] -> [p] : 0 <= p < 8 }");
    CHECK(bool(A) && bool(P), "构造 A 与 P");

    isl_map *Pinv = isl_map_reverse(P.release());
    ISLMap dep(isl_map_apply_range(A.release(), Pinv));
    CHECK(bool(dep), "Dep = A . P^-1 求值成功");

    // 期望：{ [m,n] -> [m] }，即每个消费者 task 只等 1 个生产者 → wait = 1
    ISLSet oneTask(isl_set_read_from_str(ctx.get(), "{ [3,2] }"));
    isl_set *producers =
        isl_set_apply(oneTask.release(), isl_map_copy(dep.get()));
    // isl_set_count_val 是 __isl_keep：只借用，producers 仍归我们释放。
    isl_val *card = isl_set_count_val(producers);
    long waitCount = isl_val_get_num_si(card);
    isl_val_free(card);
    isl_set_free(producers);
    std::printf("      单个消费者 task 的 wait count = %ld\n", waitCount);
    CHECK(waitCount == 1, "wait == 1（骨架 §3.6 边 1 的期望值）");
  }

  // --- 4. barvinok 参数化计数 ----------------------------------------------
  //     P3.3 要用它算「事件张量形状随符号 shape 变化」的闭式。
  std::printf("4. barvinok 参数化基数计数\n");
  {
    // task 数 N(S) = ceil(S/128)，用 { [i] : 0 <= i < S } 的势做代理演示
    ISLSet s(isl_set_read_from_str(ctx.get(),
                                   "[S] -> { [i] : 0 <= i < S }"));
    CHECK(bool(s), "解析含参数 S 的集合");

    isl_pw_qpolynomial *card = isl_set_card(s.release());
    CHECK(card != nullptr, "isl_set_card 返回分段拟多项式");
    if (card) {
      char *str = isl_pw_qpolynomial_to_str(card);
      std::printf("      |{ [i] : 0<=i<S }| = %s\n", str ? str : "?");
      // 期望形如 [S] -> { S : S > 0; 0 : S <= 0 }
      bool ok = str && std::strstr(str, "S") != nullptr;
      CHECK(ok, "结果是 S 的符号表达式而非具体数值");
      free(str);
      isl_pw_qpolynomial_free(card);
    }
  }

  std::printf("\n%s (%d 项失败)\n", failures ? "存在失败" : "全部通过",
              failures);
  return failures ? 1 : 0;
}
