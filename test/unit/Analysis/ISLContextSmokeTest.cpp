// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Smoke test for the ISL bridge layer. Covers things Phase 3 will lean on
// repeatedly:
//   1. Ownership through the RAII wrapper is correct (isl's leak check does not
//      abort when the context is destroyed)
//   2. The error policy is CONTINUE, not abort -- malformed input must not kill
//      the process
//   3. barvinok's parametric cardinality counting works -- the basis for
//      computing wait / fan-out in P3.3

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
      std::printf("  [FAIL] %s\n", msg);                                       \
      ++failures;                                                              \
    } else {                                                                   \
      std::printf("  [ OK ] %s\n", msg);                                       \
    }                                                                          \
  } while (0)

int main() {
  ISLContext ctx;

  // --- 1. Basic parsing and move semantics ---------------------------------
  std::printf("1. parsing and RAII\n");
  {
    ISLMap m = ctx.parseMap("{ [i,j] -> [i] : 0 <= i < 128 and 0 <= j < 64 }");
    CHECK(bool(m), "parse an affine map");

    ISLMap moved = std::move(m);
    CHECK(bool(moved) && !m, "after move the source is null and the target owns");
  } // `moved` is destroyed here; on an ownership bug isl aborts when the
    // context is freed

  // --- 2. Error policy: malformed input must not abort ---------------------
  std::printf("2. error handling\n");
  {
    ISLMap bad = ctx.parseMap("this is not valid isl syntax {{{");
    CHECK(!bad, "malformed input returns empty rather than crashing");
    CHECK(ctx.hasError(), "the error state is recorded");
    ctx.clearError();
    CHECK(!ctx.hasError(), "the error can be cleared");
  }

  // --- 3. Minimal walkthrough of Dep = P^-1 . A ----------------------------
  //     Skeleton section 3.6, edge 1 (RMSNorm1 -> Wq): consumer task (m,n)
  //     depends on producer task m.
  std::printf("3. dependence derivation Dep = P^-1 . A\n");
  {
    // A_cons: consumer task coordinate -> the row block of elements it reads
    ISLMap A = ctx.parseMap("{ [m,n] -> [m] : 0 <= m < 8 and 0 <= n < 4 }");
    // P_prod: producer task coordinate -> the row block it writes (identity
    // here)
    ISLMap P = ctx.parseMap("{ [p] -> [p] : 0 <= p < 8 }");
    CHECK(bool(A) && bool(P), "construct A and P");

    isl_map *Pinv = isl_map_reverse(P.release());
    ISLMap dep(isl_map_apply_range(A.release(), Pinv));
    CHECK(bool(dep), "Dep = A . P^-1 evaluates");

    // Expected: { [m,n] -> [m] }, i.e. each consumer task waits on exactly one
    // producer, so wait = 1
    ISLSet oneTask(isl_set_read_from_str(ctx.get(), "{ [3,2] }"));
    isl_set *producers =
        isl_set_apply(oneTask.release(), isl_map_copy(dep.get()));
    // isl_set_count_val is __isl_keep: it only borrows, so we still free
    // `producers` ourselves.
    isl_val *card = isl_set_count_val(producers);
    long waitCount = isl_val_get_num_si(card);
    isl_val_free(card);
    isl_set_free(producers);
    std::printf("      wait count for one consumer task = %ld\n", waitCount);
    CHECK(waitCount == 1, "wait == 1 (expected by skeleton section 3.6 edge 1)");
  }

  // --- 4. barvinok parametric counting -------------------------------------
  //     P3.3 uses this to derive closed forms for how event tensor shapes vary
  //     with symbolic shapes.
  std::printf("4. barvinok parametric cardinality counting\n");
  {
    // Task count N(S) = ceil(S/128); demonstrated here via the cardinality of
    // { [i] : 0 <= i < S } as a stand-in
    ISLSet s(isl_set_read_from_str(ctx.get(),
                                   "[S] -> { [i] : 0 <= i < S }"));
    CHECK(bool(s), "parse a set parameterised by S");

    isl_pw_qpolynomial *card = isl_set_card(s.release());
    CHECK(card != nullptr, "isl_set_card returns a piecewise quasipolynomial");
    if (card) {
      char *str = isl_pw_qpolynomial_to_str(card);
      std::printf("      |{ [i] : 0<=i<S }| = %s\n", str ? str : "?");
      // Expected to look like [S] -> { S : S > 0; 0 : S <= 0 }
      bool ok = str && std::strstr(str, "S") != nullptr;
      CHECK(ok, "the result is a symbolic expression in S, not a number");
      free(str);
      isl_pw_qpolynomial_free(card);
    }
  }

  std::printf("\n%s (%d failed)\n", failures ? "FAILURES PRESENT" : "ALL PASSED",
              failures);
  return failures ? 1 : 0;
}
