#!/usr/bin/env python3
"""Run the specified serving GEMM shape/config/split/epilogue matrix against torch."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/experiments/SERVING_R10/task_body_tests/gemm_matrix"
WORK = Path("/root/r10_work/gemm_matrix")
BINARY = Path("/root/r10_work/serving_gemm_matrix_test")
CONFIGS = [
    (16, 128, 64, 2), (16, 128, 64, 4), (16, 128, 128, 2),
    (16, 64, 128, 3), (32, 128, 64, 3), (64, 128, 64, 3),
    (128, 128, 64, 2),
]
SHAPES = [(n, k) for n, k in ((64, 2048), (3072, 2048),
                              (2048, 8192), (256, 6144))]
ROWS = (1, 3, 16, 17, 64, 1024)
SPLITS = (1, 4)
OPS = ("store", "residual", "swiglu", "argmax_partial")


def ordered_bf16(value: np.ndarray) -> np.ndarray:
    bits = value.view(np.uint16).astype(np.int32)
    return np.where(bits & 0x8000, 0xFFFF - bits, 0x8000 + bits)


def expected_values(acc: torch.Tensor, operation: str,
                    tile_n: int) -> tuple[np.ndarray, np.ndarray | None]:
    rounded = acc.to(torch.bfloat16)
    if operation == "store":
        return rounded.view(torch.uint16).cpu().numpy(), None
    if operation == "residual":
        value = (rounded.float() + 0.125).to(torch.bfloat16)
        return value.view(torch.uint16).cpu().numpy(), None
    if operation == "swiglu":
        group = acc.reshape(acc.shape[0], acc.shape[1] // 32, 2, 16)
        gate = group[:, :, 0, :].to(torch.bfloat16).float()
        up = group[:, :, 1, :].to(torch.bfloat16).float()
        silu = (gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16)
        value = (silu.float() * up).to(torch.bfloat16)
        return value.reshape(acc.shape[0], -1).view(torch.uint16).cpu().numpy(), None
    if operation == "argmax_partial":
        values, indices = [], []
        for start in range(0, acc.shape[1], tile_n):
            slice_ = rounded[:, start:start + tile_n].float()
            maximum = slice_.max(dim=1).values
            index = (slice_ == maximum[:, None]).int().argmax(dim=1) + start
            values.append(maximum)
            indices.append(index)
        return (torch.stack(values, 1).cpu().numpy().astype(np.float32),
                torch.stack(indices, 1).cpu().numpy().astype(np.int32))
    raise AssertionError(operation)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if not BINARY.exists():
        raise RuntimeError(f"compile test/unit/serving_gemm_matrix_test.cu first: {BINARY}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    # Candidate measurements use this lock; the full matrix must never
    # contaminate top-three timing or mix with another benchmark process.
    lock = Path("/root/r10_work/serving_gpu.lock")
    cases = 0
    failure = None
    started = time.perf_counter()
    with lock.open("w") as handle, (OUT / "cases.jsonl").open("w") as log:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for rows in ROWS:
            for columns, reduction in SHAPES:
                a_rows = torch.arange(rows, device="cuda")[:, None]
                a_k = torch.arange(reduction, device="cuda")[None, :]
                b_rows = torch.arange(columns, device="cuda")[:, None]
                b_k = torch.arange(reduction, device="cuda")[None, :]
                a = (((a_rows * 11 + a_k * 3) % 7 - 3).float() / 64).to(torch.bfloat16)
                b = (((b_rows * 13 + b_k * 2) % 5 - 2).float() / 64).to(torch.bfloat16)
                for split in SPLITS:
                    acc = torch.zeros((rows, columns), dtype=torch.float32, device="cuda")
                    for part in range(split):
                        lo = part * (reduction // split)
                        hi = (part + 1) * (reduction // split)
                        acc += a[:, lo:hi].float() @ b[:, lo:hi].float().T
                    for config, (_, tile_n, _, _) in enumerate(CONFIGS):
                        for operation, name in enumerate(OPS):
                            dump = WORK / "case.bin"
                            argv = [str(BINARY), str(config), str(rows),
                                    str(columns), str(reduction), str(split),
                                    str(operation), str(dump)]
                            process = subprocess.run(argv, capture_output=True, text=True)
                            row = {"config": config, "tile": CONFIGS[config],
                                   "m": rows, "n": columns, "k": reduction,
                                   "split": split, "op": name,
                                   "returncode": process.returncode}
                            if process.returncode:
                                row["stderr"] = process.stderr[-1000:]
                            else:
                                expected, expected_index = expected_values(acc, name, tile_n)
                                if expected_index is None:
                                    actual = np.fromfile(dump, dtype=np.uint16).reshape(expected.shape)
                                    delta = np.abs(ordered_bf16(actual) - ordered_bf16(expected))
                                    row["max_ulp"] = int(delta.max())
                                    row["violations"] = int(np.count_nonzero(delta > 1))
                                    row["elements"] = int(delta.size)
                                else:
                                    count = expected.size
                                    raw = dump.read_bytes()
                                    actual = np.frombuffer(raw[:4 * count], dtype=np.float32).reshape(expected.shape)
                                    indices = np.frombuffer(raw[4 * count:], dtype=np.int32).reshape(expected.shape)
                                    row["max_value_error"] = float(np.max(np.abs(actual - expected)))
                                    row["index_mismatches"] = int(np.count_nonzero(indices != expected_index))
                                    row["violations"] = int(np.count_nonzero(actual != expected)) + row["index_mismatches"]
                                    row["elements"] = int(count)
                            row["pass"] = process.returncode == 0 and row.get("violations", 1) == 0
                            log.write(json.dumps(row) + "\n")
                            log.flush()
                            cases += 1
                            if not row["pass"]:
                                failure = row
                                break
                        if failure:
                            break
                    if failure:
                        break
                if failure:
                    break
            if failure:
                break
    summary = {"cases": cases, "expected_cases": len(CONFIGS) * len(ROWS) *
               len(SHAPES) * len(SPLITS) * len(OPS),
               "seconds": time.perf_counter() - started, "pass": failure is None,
               "first_failure": failure}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
