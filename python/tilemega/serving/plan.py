"""Versioned ctypes binding for one generated TileMega serving plan."""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from pathlib import Path


ABI_VERSION = 1


class PlanInfo(C.Structure):
    _fields_ = [
        ("abi_version", C.c_uint32), ("phase", C.c_uint32),
        ("batch_lo", C.c_int32), ("batch_hi", C.c_int32),
        ("seq", C.c_int32), ("past_lo", C.c_int32),
        ("past_hi", C.c_int32), ("capacity", C.c_int32),
        ("grid", C.c_int32), ("residency", C.c_int32),
        ("modes", C.c_uint32), ("buffer_count", C.c_uint32),
    ]


class BufferInfo(C.Structure):
    _fields_ = [
        ("name", C.c_char_p), ("role", C.c_uint32), ("dtype", C.c_uint32),
        ("elements_constant", C.c_uint64),
        ("elements_per_batch", C.c_uint64), ("pack_json", C.c_char_p),
    ]


@dataclass(frozen=True)
class Buffer:
    name: str
    role: int
    dtype: int
    elements_constant: int
    elements_per_batch: int
    pack_json: str


class PlanLibrary:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        self.lib = C.CDLL(str(self.path), mode=C.RTLD_LOCAL)
        self.lib.tm_plan_query.argtypes = [C.POINTER(PlanInfo)]
        self.lib.tm_plan_query.restype = C.c_int
        self.lib.tm_plan_buffer.argtypes = [C.c_uint32, C.POINTER(BufferInfo)]
        self.lib.tm_plan_buffer.restype = C.c_int
        self.lib.tm_plan_create.argtypes = [C.c_int, C.POINTER(C.c_void_p), C.c_int]
        self.lib.tm_plan_create.restype = C.c_void_p
        self.lib.tm_plan_set_steps.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_uint32]
        self.lib.tm_plan_set_steps.restype = C.c_int
        self.lib.tm_plan_launch.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32,
                                            C.c_uint64, C.c_void_p]
        self.lib.tm_plan_launch.restype = C.c_int
        self.lib.tm_plan_destroy.argtypes = [C.c_void_p]
        self.lib.tm_plan_destroy.restype = None
        info = PlanInfo()
        if self.lib.tm_plan_query(C.byref(info)) != 0 or info.abi_version != ABI_VERSION:
            raise RuntimeError(f"incompatible serving plan ABI: {path}")
        self.info = info
        buffers = []
        for index in range(info.buffer_count):
            item = BufferInfo()
            if self.lib.tm_plan_buffer(index, C.byref(item)) != 0:
                raise RuntimeError(f"buffer {index} could not be queried")
            buffers.append(Buffer(
                item.name.decode(), item.role, item.dtype,
                item.elements_constant, item.elements_per_batch,
                item.pack_json.decode() if item.pack_json else "",
            ))
        self.buffers = tuple(buffers)

    def create(self, batch: int, pointers: dict[str, int], device: int) -> "Plan":
        if not self.info.batch_lo <= batch <= self.info.batch_hi:
            raise ValueError("batch outside the solved interval")
        addresses = (C.c_void_p * len(self.buffers))()
        for index, buffer in enumerate(self.buffers):
            if buffer.role == 1:
                if buffer.name not in pointers:
                    raise KeyError(f"missing external buffer: {buffer.name}")
                addresses[index] = pointers[buffer.name]
            elif buffer.name in pointers:
                raise ValueError(f"internal buffer supplied externally: {buffer.name}")
        ptr = self.lib.tm_plan_create(batch, addresses, device)
        if not ptr:
            raise RuntimeError("tm_plan_create rejected the plan or its buffers")
        return Plan(self, ptr)


class Plan:
    def __init__(self, library: PlanLibrary, handle: int):
        self.library = library
        self.handle = handle
        self.iteration = {1: 0, 2: 0}

    def set_steps(self, past: list[int]) -> None:
        if not past or any(not self.library.info.past_lo <= p <= self.library.info.past_hi
                           for p in past):
            raise ValueError("past outside the solved interval")
        values = (C.c_int32 * len(past))(*past)
        if self.library.lib.tm_plan_set_steps(self.handle, values, len(past)) != 0:
            raise RuntimeError("tm_plan_set_steps failed")

    def launch(self, step: int, mode: int, stream: int) -> None:
        if not self.library.info.modes & mode:
            raise ValueError("mode not present in this plan")
        if self.library.lib.tm_plan_launch(
                self.handle, step, mode, self.iteration[mode], stream) != 0:
            raise RuntimeError("tm_plan_launch failed")
        self.iteration[mode] += 1

    def close(self) -> None:
        if self.handle:
            self.library.lib.tm_plan_destroy(self.handle)
            self.handle = 0

    def __enter__(self) -> "Plan":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
