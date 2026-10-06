"""Reusable bounded native scratch; no new host registration or graph mapping."""
import ctypes as C
import numpy as np
from candidates.ukl_native_sampling_v10 import sampling as S

RETAINED = []
CAPACITY = 32 * 1024**2


class Workspace:
    def __init__(self, api, owner):
        self.api, self.owner = api, owner
        self.device = C.c_void_p()
        self.host = np.empty(CAPACITY, dtype=np.uint8)
        self.closed = self.retained = False
        api.call('cudaMalloc', C.byref(self.device), CAPACITY)
        if not self.device.value:
            raise RuntimeError('Null native workspace')

    def retain(self):
        if not self.retained:
            self.retained = True
            RETAINED.append(self)

    def run(self, fn, view, seeds, fanout, grouped, seed, out):
        if self.closed or self.retained:
            raise RuntimeError('Closed or retained workspace')
        begin = (seeds.nbytes + 7) // 8 * 8
        cursor = begin
        offsets = {}
        for key, value in out.items():
            cursor = (cursor + 7) // 8 * 8
            offsets[key] = cursor
            cursor += value.nbytes
        if cursor > CAPACITY:
            raise ValueError('Native workspace exceeds fixed 32 MiB budget')
        base = self.device.value
        try:
            self.api.call('cudaMemcpy', base, seeds.ctypes.data, seeds.nbytes, 1)
            self.api.call('cudaMemset', base + begin, 255, cursor - begin)
            code = fn(view, C.c_void_p(base), len(seeds), fanout, int(grouped), seed,
                      S.Out(*[base + offsets[key] for key in out]))
            self.api.synchronize()
            # One compact D2H transfer per layer, preserving all old validation.
            self.api.call('cudaMemcpy', self.host.ctypes.data + begin,
                          base + begin, cursor - begin, 2)
            for key, value in out.items():
                value[:] = np.ndarray(value.shape, dtype=value.dtype, buffer=self.host,
                                      offset=offsets[key])
            return code
        except BaseException:
            # Any uncertain CUDA state retains the graph/registration owner.
            self.retain()
            raise

    def close(self):
        if self.closed:
            return
        if self.retained:
            raise RuntimeError('Retained CUDA workspace: keep graph registered until process exit')
        try:
            self.api.synchronize()
            self.api.call('cudaFree', self.device)
        except BaseException:
            self.retain()
            raise
        self.closed = True
        self.host = self.owner = None


class Native(S.Native):
    def __init__(self, graph, registration=None):
        super().__init__(graph, registration)
        self.workspace = None

    def _cuda(self, native_view, seeds, fanout, grouped, seed, out):
        if self.workspace is None:
            self.workspace = Workspace(self.registration.api, self)
        try:
            return self.workspace.run(self.fn, native_view, seeds, fanout, grouped, seed, out)
        except BaseException:
            if self.workspace.retained:
                self.retained = True
            raise

    def close(self):
        if self.workspace is not None:
            self.workspace.close()
        super().close()
