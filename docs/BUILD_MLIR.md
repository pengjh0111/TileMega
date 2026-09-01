# Building the MLIR dependency

The Coupling Graph dialect is the sole L4→L1 contract (skeleton §2.6), so MLIR is
a hard build dependency: `TILEMEGA_ENABLE_MLIR` defaults to `ON` and CMake fails
rather than silently degrading when the package is missing.

- Pinned LLVM commit: `23a60f15f2fcafcf67b95b0a035053579958b732`
  (`third_party/cutlass/cutlass_compiler/LLVM_COMMIT`)
- Package version produced: `23.0.0git`

## Build and install

```sh
scripts/build_mlir.sh /path/to/prefix          # clone, build, install
cmake -S . -B build -G Ninja \
      -DMLIR_DIR=/path/to/prefix/lib/cmake/mlir
```

`scripts/build_mlir.sh` builds only `Native;NVPTX`, `LLVM_ENABLE_PROJECTS=mlir`,
`Release`, and then **installs**. The install step is not optional — see below.

## Why an install tree and not a build tree

An LLVM/MLIR *build tree* exports a CMake package that is **not relocatable**.
`MLIRConfig.cmake` bakes absolute paths to both the build directory and the
original source directory, because MLIR's headers live in both places
(hand-written headers in the source tree, TableGen `.inc` output in the build
tree):

```cmake
set(MLIR_INCLUDE_DIRS "<source>/mlir/include;<build>/llvm-project/tools/mlir/include")
```

`LLVMConfig.cmake` does the same for `LLVM_INCLUDE_DIRS`, `LLVM_LIBRARY_DIRS`,
`LLVM_BUILD_MAIN_SRC_DIR` and `LLVM_TOOLS_BINARY_DIR`, and `MLIRTargets.cmake` /
`LLVMExports.cmake` record every imported library by absolute path.

Move or rename either directory afterwards and every one of those paths dangles.
An *install tree* computes its paths relative to the config file's own location,
so it survives being moved. Consume an install tree.

## Do not bridge a moved build tree with symlinks

If a build tree has already been moved, it is tempting to recreate the old paths
as symlinks so the baked paths resolve again. Do not put such a symlink inside
`third_party/`: a path inside a submodule that is not part of that submodule's
tracked content leaves the submodule permanently dirty, is invisible to every
other checkout, and makes the build depend on one machine's history.

To recover a moved build tree without touching the repository, copy its two CMake
package directories to a local prefix and rewrite the baked paths there:

```sh
PREFIX=/path/to/prefix
mkdir -p "$PREFIX/lib/cmake"
cp -a <build>/lib/cmake/mlir              "$PREFIX/lib/cmake/mlir"
cp -a <build>/llvm-project/lib/cmake/llvm "$PREFIX/lib/cmake/llvm"
# point the package's self-references at $PREFIX, and every other reference at
# the build tree's and source tree's current locations
```

Five files carry the absolute paths: `MLIRConfig.cmake`, `MLIRTargets.cmake`,
`LLVMConfig.cmake`, `LLVMExports.cmake`, `LLVMBuildTreeOnlyTargets.cmake`.
This is a local environment fix; it belongs in the environment, never in a commit.

## lit

`check-cg-lit` locates `lit.py` from `LLVM_BUILD_MAIN_SRC_DIR`, exported by
`LLVMConfig`. Override with `-DTILEMEGA_LIT_DRIVER=/path/to/llvm/utils/lit/lit.py`
if you consume an MLIR package that does not export it.
