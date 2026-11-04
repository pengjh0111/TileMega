#!/usr/bin/env python3
"""Apply the pinned LLVM descriptor API compatibility patch and rebuild TVM.

This CPU-only step reuses the configured CUDA/cuBLAS/cuDNN/LLVM build. It does
not install packages, initialize CUDA, or replace an earlier build receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


TVM_COMMIT = "7b5acb32bd271342a24130f4ef439ff6c2e52c6c"
SOURCE_FILE = "src/target/llvm/llvm_instance.cc"
BEFORE_SHA256 = "b5613537b1d55b9e7398557ba07eeb850bb73422409e5aa2f399ea7a1d76eda7"
AFTER_SHA256 = "aa429c6c4fbf853f748fb8e99f747c5bfea94a59de8f445a001f92787ab9041d"
BUILDINFO_FILE = "python/tvm/support/libinfo.py"
BUILDINFO_BEFORE = "c12245def0db46c5700cd8e5924027d12a96e3ed1298830de5c07cfc691f5b88"
BUILDINFO_AFTER = "33f6821a23203be6da8cfb48eaaf215c3f027cae52a0ee6028b6eebde0fc007c"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo / "runs/dnn1/tvm/source")
    parser.add_argument("--build", type=Path, default=repo / "runs/dnn1/tvm/build")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=14400)
    parser.add_argument("--python", default="/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python")
    options = parser.parse_args()
    if options.jobs < 1 or not 1 <= options.timeout_s <= 14400:
        parser.error("invalid CPU jobs or wall limit")
    source, build, out = (value.resolve() for value in
                          (options.source, options.build, options.out))
    out.mkdir(parents=True, exist_ok=False)
    patch = Path(__file__).with_name("dnn1_tvm_llvm23_compat.patch")
    source_file = source / SOURCE_FILE
    receipt = dict(schema_version=1, evidence="verified", state="preparing",
                   start_utc=utc(), source=str(source), build=str(build),
                   gpu_initialization=False, wall_limit_s=options.timeout_s,
                   patch=str(patch), patch_sha256=sha(patch))

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    started = time.monotonic()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                         text=True).strip()
        if commit != TVM_COMMIT:
            raise ValueError(f"unexpected TVM commit {commit}")
        receipt["source_commit"] = commit
        original = subprocess.check_output(["git", "-C", str(source), "show",
                                            f"HEAD:{SOURCE_FILE}"])
        if hashlib.sha256(original).hexdigest() != BEFORE_SHA256:
            raise ValueError("unexpected unpatched TVM source hash")
        (out / "llvm_instance.original.cc").write_bytes(original)
        receipt["source_original_sha256"] = BEFORE_SHA256
        receipt["source_at_entry_sha256"] = sha(source_file)
        if receipt["source_at_entry_sha256"] == BEFORE_SHA256:
            subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
            receipt["patch_applied_by_script"] = True
        elif receipt["source_at_entry_sha256"] == AFTER_SHA256:
            receipt["patch_applied_by_script"] = False
        else:
            raise ValueError("source is neither the pinned original nor the exact patched file")
        if sha(source_file) != AFTER_SHA256:
            raise ValueError("patched TVM source hash differs")
        receipt["source_patched_sha256"] = AFTER_SHA256
        shutil.copy2(source_file, out / "llvm_instance.patched.cc")
        shutil.copy2(patch, out / patch.name)
        buildinfo_patch = Path(__file__).with_name("dnn1_tvm_buildinfo.patch")
        buildinfo_source = source / BUILDINFO_FILE
        original_info = subprocess.check_output(["git", "-C", str(source), "show",
                                                 f"HEAD:{BUILDINFO_FILE}"])
        if hashlib.sha256(original_info).hexdigest() != BUILDINFO_BEFORE:
            raise ValueError("unexpected unpatched TVM build-info source hash")
        (out / "libinfo.original.py").write_bytes(original_info)
        if sha(buildinfo_source) == BUILDINFO_BEFORE:
            subprocess.run(["git", "-C", str(source), "apply", "--check", str(buildinfo_patch)],
                           check=True)
            subprocess.run(["git", "-C", str(source), "apply", str(buildinfo_patch)], check=True)
        if sha(buildinfo_source) != BUILDINFO_AFTER:
            raise ValueError("TVM build-info source is neither the original nor exact patched file")
        shutil.copy2(buildinfo_source, out / "libinfo.patched.py")
        shutil.copy2(buildinfo_patch, out / buildinfo_patch.name)
        receipt["buildinfo_patch"] = dict(path=str(buildinfo_patch), sha256=sha(buildinfo_patch),
                                          original_source_sha256=BUILDINFO_BEFORE,
                                          patched_source_sha256=BUILDINFO_AFTER)
        dtype_patch = Path(__file__).with_name("dnn1_tvm_torch_dtype.patch")
        applied = subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check",
                                  str(dtype_patch)], capture_output=True).returncode == 0
        if not applied:
            subprocess.run(["git", "-C", str(source), "apply", "--check", str(dtype_patch)], check=True)
            subprocess.run(["git", "-C", str(source), "apply", str(dtype_patch)], check=True)
        shutil.copy2(dtype_patch, out / dtype_patch.name)
        receipt["torch_dtype_patch"] = dict(path=str(dtype_patch), sha256=sha(dtype_patch),
                                            applied_by_script=not applied)
        failure = repo / "runs/dnn1/tvm/build_failure.json"
        receipt["original_failure"] = dict(path=str(failure), sha256=sha(failure))
        receipt["cmake_cache_sha256"] = sha(build / "CMakeCache.txt")
        cache = (build / "CMakeCache.txt").read_text().splitlines()
        receipt["configured_features"] = [line for line in cache if any(
            line.startswith(key + ":") for key in
            ("USE_CUDA", "USE_CUBLAS", "USE_CUDNN", "USE_LLVM", "CMAKE_CUDA_ARCHITECTURES",
             "CUDA_CUDNN_LIBRARY"))]

        llvm_prefix = Path("/root/TileMega/runs/r16/mlir-install")
        llvm_header = llvm_prefix / "include/llvm/MC/MCSubtargetInfo.h"
        receipt["llvm_header"] = dict(path=str(llvm_header), sha256=sha(llvm_header))
        helper = source_file.read_text().split(
            "// LLVM 23 snapshots expose either key() or Key under the same version macro.", 1
        )[1].split("namespace defaults {", 1)[0]
        probe = out / "llvm_key_probe.cpp"
        probe.write_text("#include <llvm/MC/MCSubtargetInfo.h>\n#include <type_traits>\n"
                         "#include <utility>\n" + helper + """
struct OldSnapshot { const char* Key; };
struct NewSnapshot { llvm::StringRef key() const; };
static_assert(std::is_same_v<decltype(SubtargetKey(std::declval<OldSnapshot>(), 0)), const char*>);
static_assert(std::is_same_v<decltype(SubtargetKey(std::declval<NewSnapshot>(), 0)), llvm::StringRef>);
static_assert(std::is_constructible_v<llvm::StringRef, decltype(SubtargetKey(std::declval<llvm::SubtargetSubTypeKV>(), 0))>);
static_assert(std::is_constructible_v<llvm::StringRef, decltype(SubtargetKey(std::declval<llvm::SubtargetFeatureKV>(), 0))>);
""")
        command = ["/usr/bin/c++", "-std=c++20", "-I" + str(llvm_prefix / "include"),
                   "-fsyntax-only", str(probe)]
        receipt["probe_command"] = command
        with (out / "llvm_key_probe.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        receipt["probe_passed"] = True
        command = ["/root/.venvs/mpk/bin/cmake", "--build", str(build), "-j", str(options.jobs)]
        receipt.update(state="building", build_command=command)
        save()
        with (out / "build.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                result = process.wait(timeout=options.timeout_s)
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    result = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    result = process.wait()
        receipt.update(state="built" if result == 0 else "build_failed", exit_code=result)
        receipt["libraries"] = [dict(path=str(path), sha256=sha(path))
                                for path in sorted((build / "lib").glob("*.so*"))]
        if result:
            raise subprocess.CalledProcessError(result, command)
        # The compiler's C++ FFI headers and Python extension must come from
        # the same pinned submodule. A newer PyPI wheel can load successfully
        # yet crash when serializing IR during MetaSchedule initialization.
        ffi_source = source / "3rdparty/tvm-ffi"
        wheel_dir = out / "ffi-wheels"
        wheel_dir.mkdir()
        ffi_env = dict(env, CMAKE_BUILD_PARALLEL_LEVEL=str(options.jobs))
        wheel_command = [options.python, "-m", "pip", "wheel", "--no-deps",
                         "--wheel-dir", str(wheel_dir), str(ffi_source)]
        with (out / "ffi_build.log").open("w") as log:
            subprocess.run(wheel_command, stdout=log, stderr=subprocess.STDOUT,
                           env=ffi_env, check=True,
                           timeout=max(1, options.timeout_s-(time.monotonic()-started)))
        wheel, = wheel_dir.glob("apache_tvm_ffi-*.whl")
        destination = source.parent / "ffi_python"
        install_command = [options.python, "-m", "pip", "install", "--no-deps", "--no-index",
                           "--upgrade", "--target", str(destination), str(wheel)]
        with (out / "ffi_install.log").open("w") as log:
            subprocess.run(install_command, stdout=log, stderr=subprocess.STDOUT,
                           env=ffi_env, check=True,
                           timeout=max(1, options.timeout_s-(time.monotonic()-started)))
        receipt["python_ffi"] = dict(
            source_commit=subprocess.check_output(["git", "-C", str(ffi_source),
                                                    "rev-parse", "HEAD"], text=True).strip(),
            wheel=str(wheel), wheel_sha256=sha(wheel), target=str(destination),
            build_command=wheel_command, install_command=install_command)
    except Exception as error:
        receipt.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(end_utc=utc(), elapsed_s=time.monotonic() - started)
        save()


if __name__ == "__main__":
    main()
