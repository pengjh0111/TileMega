# External dependencies: isl / barvinok

Evidence labels: ✅ compiled/executed and observed; ⚠️ documented but
unconfirmed; ❌ conjecture.

This records Part 1's dependency-feasibility work for the ISL/barvinok
migration (skeleton principle 3: "CuTe is representation, ISL is solver").
The build recipe here is what `cmake/ISL.cmake` expects to find.

## 1. Answers to the required questions

**(1) Do isl/barvinok conflict with MLIR? Evidence.**

✅ No. `docs/experiments/P3_ISL/crosslink_probe.cpp` builds an
`mlir::MLIRContext`/`mlir::ModuleOp`, exercises MLIR's own bundled
`mlir::presburger::IntegerRelation` (`isIntegerEmpty()` on `{[i]:0<=i<10}`),
builds an `isl_set` via the isl C API, computes its cardinality with
barvinok's `isl_set_card` (a piecewise quasi-polynomial), and checks a subset
relation with `isl_set_is_subset` — all in one process. Linked as a CMake
target (`isl_crosslink_test`, registered as ctest `isl_crosslink`) against
`MLIRIR`, `MLIRSupport`, `MLIRPresburger`, and the isl/polylib/barvinok/
ntl/gmp static libraries. Result:

```
[mlir-presburger] { [i] : 0<=i<10 } isIntegerEmpty=0
[isl+barvinok] card([n]->{[i]:0<=i<n}) = [n] -> { n : n > 0 }
[isl] {[i]:0<=i<5} subset of {[i]:0<=i<10} = 1
RESULT mlir_ok=1 isl_barvinok_ok=1
```

`ctest -R isl_crosslink`: 1/1 passed. No symbol collisions, no crashes, no
silent corruption. This confirms the task's stated premise: barvinok depends
on isl and GMP, not on LLVM, and MLIR's Presburger implementation is a
separate, LLVM-`APInt`/`DynamicAPInt`-backed piece of code in a different
namespace (`mlir::presburger::*` vs. isl's C `isl_*` symbols) — the two
systems are orthogonal at both the symbol and the semantic level, and nothing
here forces them to agree.

✅ Confirmed independently: `nm` on every static library under the pinned
MLIR build (`/tmp/tilemega-vf-build/llvm-project/lib/*.a`, see
`docs/experiments/V_F/result.md` for how that tree is built) finds zero
`__gmp*` symbol references. MLIR does not link GMP at all.

**(2) Which integration approach, and why?**

Approach 2 from the task's priority list: **system GMP + an out-of-tree
autotools build**, matching this repo's existing convention for MLIR itself
(build once outside the CMake tree, point CMake at the result via a
`_DIR`/`_BUILD_DIR` cache variable — see `docs/BUILD_MLIR.md` and the
`-DMLIR_DIR=` pattern already used for MLIR). `cmake/ISL.cmake` does not use
`ExternalProject_Add`: autoreconf/libtool integration inside a CMake
superbuild is its own source of fragility, and this repo already has a
working precedent for the simpler "build once, `find_library`/`find_path` the
result" model.

Why not `ExternalProject_Add`: no functional advantage here — none of these
three projects use CMake, so a superbuild step would still shell out to
`autogen.sh`/`configure`/`make`, just from inside a CMake custom command
instead of a documented recipe. It would also complicate incremental
rebuilds and reconfiguration (autotools build directories do not play well
with being deleted/regenerated automatically), for no gain, since these
libraries change essentially never once configured for this target.

**(3) Which integer backend did isl use — imath or GMP?**

**GMP**, not imath, despite the task's priority list ranking imath first.
Reasoning, in order:

- (1) was answered first and shows there is no real GMP/MLIR conflict to
  avoid — the entire reason imath was the preferred first option ("消除 GMP
  依赖" if GMP conflicts) does not apply here.
- `polylib` — a hard, non-optional dependency of barvinok (`AX_SUBMODULE
  (polylib,build|bundled|system,bundled)` has no `no` choice, unlike `pet`)
  — only ships a GMP-flavored build in this integration
  (`barvinok`'s own `configure.ac` hardcodes `POLYLIB_LIBS="-lpolylibgmp"`
  and the bundled path always builds `libpolylibgmp`). So barvinok's overall
  dependency chain needs GMP for polylib *regardless* of what isl itself
  uses.
- Given GMP is unavoidable via polylib, building isl with GMP too (rather
  than isl+imath alongside polylib+GMP) keeps exactly one bignum backend in
  the process instead of two, which is strictly simpler and removes a
  category of "which representation is this `Value` in" bugs at zero cost,
  since the premise for avoiding GMP (a conflict with MLIR) does not hold.

isl's own `imath` submodule was still fetched (`third_party/barvinok/isl/
imath`, pinned via isl's own `.gitmodules`) and is available if a future
target genuinely cannot link GMP, but it is not used by the configuration
`cmake/ISL.cmake` builds against.

**(4) When MLIR's Presburger implementation and isl coexist, who is the
semantic authority?**

**isl**, unconditionally, per the task's instruction. As of this round,
`mlir::Analysis::Presburger` is used **nowhere** in TileMega's own source —
grep confirms no include of an MLIR Presburger header outside this probe file
(`crosslink_probe.cpp`, which exists only to *prove* the two coexist, not to
use MLIR's Presburger for anything semantic). The CG dialect's own verifiers
(`lib/Dialect/CouplingGraph/*Verifier*`, `lib/Analysis/*`) currently do not
touch MLIR's Presburger types at all — they either use the pre-migration
`ClosedForm`/`AffineRelation` types (see skeleton §1.5.1 and
`docs/PROPOSED_SKELETON_CHANGES.md`) or, after the Part 3 migration, isl
types directly. There is nothing to "remove" for this round; the constraint
is recorded here so a future contributor does not reach for
`mlir::presburger::IntegerRelation` as a shortcut when isl already owns this
domain.

**(5) How much does build time increase?**

✅ Measured, clean builds, this machine (112 cores):

| Component | `make clean && make -jN` wall time |
|---|---:|
| isl 0.28 | 9.3 s |
| polylib 5.22.9 | 2.3 s |
| barvinok 0.41.9 | 7.6 s |
| **Total** | **~19 s** |

Negligible next to the MLIR/LLVM build this project already requires (which
takes on the order of tens of minutes; see `docs/experiments/V_F/result.md`).
Configuring (`autogen.sh` + `configure`) for all three the first time takes
longer (autotools/libtool bootstrapping, dominated by `libtoolize` and
`configure`'s feature probing) but is a one-time cost per checkout, not part
of the normal edit/build/test loop `cmake/ISL.cmake` is wired into.

## 2. Dependency tree (as actually built)

```
barvinok 0.41.9  →  isl 0.28 (GMP backend)
                 →  polylib 5.22.9 (libpolylibgmp, GMP backend)
                 →  NTL 11.5.1 (system, apt libntl-dev)
                 →  GMP 6.2.1 (system, apt libgmp-dev)
                 →  pet: NOT built (barvinok's own default is `no`; pet is
                    for extracting polyhedra from C source, irrelevant here)
                 →  cddlib, GLPK, TOPCOM: NOT found/used (all optional;
                    `configure` silently disables what it cannot find)
```

`isl`'s own `imath` submodule was fetched but is not built into anything
(isl is configured `--with-int=gmp`).

No component was git-cloned separately from an arbitrary ref: `isl` and
`polylib` are the exact submodule commits `third_party/barvinok` (pinned at
tag `barvinok-0.41.9`, commit `dd7e6d8`) itself records —
`git submodule update --init isl polylib` inside `third_party/barvinok`
fetches precisely the tree a `barvinok-0.41.9` release tarball would bundle,
without needing to fetch a tarball. `pet` is declared in the same
`.gitmodules` but was never initialized (not needed).

Note: `third_party/barvinok`'s own `.gitmodules` points at `git://
repo.or.cz/...`, and the `git://` protocol timed out in this environment
(likely a blocked/unavailable port 9418). `git config submodule.<name>.url
https://repo.or.cz/<name>.git` before `git submodule update --init` works
around this — same host, same content, HTTPS instead of the git protocol.

## 3. Build recipe (what `cmake/ISL.cmake` expects)

```bash
# One-time: fetch the submodules barvinok's own tree pins (matched versions,
# not independently-tracked HEADs).
cd third_party/barvinok
git config submodule.isl.url https://repo.or.cz/isl.git
git config submodule.polylib.url https://repo.or.cz/polylib.git
git submodule update --init isl polylib

# System packages this needs (Ubuntu/Debian names):
apt-get install -y autoconf automake libtool pkg-config m4 \
  libgmp-dev libntl-dev

# isl, out-of-tree, GMP backend, static+PIC (so it can link into the CUDA
# host-code static library the rest of this project builds).
mkdir -p build-isl && cd build-isl
../third_party/barvinok/isl/autogen.sh   # run once, in-tree (autotools wants this)
./../third_party/barvinok/isl/configure --with-int=gmp \
  --enable-shared=no --enable-static=yes --with-pic \
  CFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC"
make -j"$(nproc)"
cd ..

# polylib, out-of-tree, same flags. --with-libgmp needs a directory, not
# "system" (unlike isl's --with-int=gmp).
mkdir -p build-polylib && cd build-polylib
../third_party/barvinok/polylib/autogen.sh
../third_party/barvinok/polylib/configure --with-libgmp=/usr \
  --enable-shared=no --enable-static=yes --with-pic CFLAGS="-O2 -fPIC"
make -j"$(nproc)"
cd ..

# barvinok itself, pointed at both build dirs above; pet disabled (not
# needed); NTL/GMP from the system.
mkdir -p build-barvinok && cd build-barvinok
../third_party/barvinok/autogen.sh
../third_party/barvinok/configure \
  --with-isl=build --with-isl-builddir=../build-isl \
  --with-polylib=build --with-polylib-builddir=../build-polylib \
  --with-pet=no --with-gmp=system \
  --enable-shared=no --enable-static=yes --with-pic \
  CFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC"
make -j"$(nproc)"
cd ..

# Then configure TileMega itself with ISL enabled:
cmake -S . -B build-portable -DMLIR_DIR=/path/to/lib/cmake/mlir \
  -DTILEMEGA_ENABLE_ISL=ON
cmake --build build-portable --target isl_crosslink_test
ctest --test-dir build-portable -R isl_crosslink
```

`autogen.sh` for `isl` and `polylib` must be run **in-tree** (they write
`configure`, `Makefile.in`, and libtool bootstrap files next to the sources);
only the `configure`/`make` step happens out-of-tree, in the sibling
`build-*` directories. `barvinok`'s own `autogen.sh` is also run in-tree and
additionally regenerates `polylib`'s autotools files as a side effect (its
Makefile.am references polylib's tree) — this is expected, not a sign
`polylib`'s own autogen step was skipped.

## 4. Toolchain versions (for `TargetSpec` / F-13 style provenance)

| Component | Version | Source |
|---|---|---|
| isl | 0.28 | `third_party/barvinok/isl`, commit `6c2b19a` (pinned by barvinok-0.41.9) |
| polylib | 5.22.9 | `third_party/barvinok/polylib`, commit `9822337` (pinned by barvinok-0.41.9) |
| barvinok | 0.41.9 | `third_party/barvinok`, tag `barvinok-0.41.9`, commit `dd7e6d8` |
| GMP | 6.2.1+dfsg-3ubuntu1 | system (`libgmp-dev`) |
| NTL | 11.5.1-1 | system (`libntl-dev`) |
| gcc/g++ | 11.4.0 | system |

## 5. What this does not cover yet

This document is Part 1 only: it establishes that the dependency stack
builds, links, and runs correctly alongside MLIR, and gives the concrete
recipe and version pins. It does not itself migrate any TileMega analysis
code off `ClosedForm`/`AffineRelation` — that is Part 3, tracked separately
in `TileMega_skeleton.md` and `docs/experiments/P3_ISL/`.
