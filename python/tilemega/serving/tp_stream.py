"""Own one MACA nonblocking stream without initializing a PyTorch stream pool."""
from __future__ import annotations

import ctypes as C
from functools import lru_cache

import torch


class MacaStreamRuntime:
    library_path = '/opt/maca/lib/libmcruntime.so'

    def __init__(self):
        self.library = C.CDLL(self.library_path, mode=C.RTLD_LOCAL)
        for name, arguments in (
                ('mcSetDevice', [C.c_int]),
                ('mcStreamCreateWithFlags', [C.POINTER(C.c_void_p), C.c_uint]),
                ('mcStreamDestroy', [C.c_void_p]),
                ('mcStreamGetFlags', [C.c_void_p, C.POINTER(C.c_uint)]),
                ('mcStreamGetPriority', [C.c_void_p, C.POINTER(C.c_int)])):
            symbol = getattr(self.library, name)
            symbol.argtypes, symbol.restype = arguments, C.c_int

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f'{operation} failed with MACA error {code}')

    def create(self, device):
        self.check(self.library.mcSetDevice(device), 'mcSetDevice')
        handle = C.c_void_p()
        self.check(self.library.mcStreamCreateWithFlags(C.byref(handle), 1), 'mcStreamCreateWithFlags')
        if not handle.value:
            raise RuntimeError('mcStreamCreateWithFlags returned a null stream')
        return handle.value

    def destroy(self, device, handle):
        self.check(self.library.mcSetDevice(device), 'mcSetDevice')
        self.check(self.library.mcStreamDestroy(C.c_void_p(handle)), 'mcStreamDestroy')

    def properties(self, handle):
        flags, priority = C.c_uint(), C.c_int()
        self.check(self.library.mcStreamGetFlags(C.c_void_p(handle), C.byref(flags)), 'mcStreamGetFlags')
        self.check(self.library.mcStreamGetPriority(C.c_void_p(handle), C.byref(priority)), 'mcStreamGetPriority')
        return dict(flags=flags.value, priority=priority.value)


@lru_cache(maxsize=1)
def runtime():
    return MacaStreamRuntime()


class OwnedTpStream:
    def __init__(self, device, *, sdk=None, wrap=None):
        self.device = int(device)
        self.sdk = sdk if sdk is not None else runtime()
        self.handle = None
        with torch.cuda.device(self.device):
            self.handle = self.sdk.create(self.device)
            try:
                self.stream = (wrap or torch.cuda.ExternalStream)(self.handle, device=self.device)
                if self.stream.cuda_stream != self.handle:
                    raise RuntimeError('ExternalStream changed the SDK stream handle')
            except BaseException:
                handle, self.handle = self.handle, None
                self.sdk.destroy(self.device, handle)
                raise

    def close(self):
        if self.handle is None:
            return
        with torch.cuda.device(self.device):
            self.stream.synchronize()
            handle, self.handle = self.handle, None
            # An SDK destroy error must not cause a second destroy submission.
            self.sdk.destroy(self.device, handle)

    def abandon(self):
        # A poisoned TP cohort can retain live peer accesses until process exit.
        self.handle = None
