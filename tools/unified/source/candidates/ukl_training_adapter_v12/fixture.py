"""Tiny owned anonymous graph lifecycle for CPU integration tests only."""
import hashlib
import tempfile
from pathlib import Path
from candidates.ukl_real_multi_v9.memory import Arena, identity
from candidates.ukl_native_sampling_v10r4 import sampling as S, fork_guard as H


class OwnedCPU:
    def __init__(self, arrays, metadata):
        if metadata['nodes'] > 4096 or sum(a.nbytes for a in arrays.values()) > 2**20:
            raise ValueError('Tiny graph fixture only')
        self.temp = tempfile.TemporaryDirectory(prefix='ukl-v12-')
        self.arena = self.graph = self.native = None
        self.guard = H.OwnershipGuard()
        specs = {}
        for key, name in S._NAMES.items():
            path = Path(self.temp.name, name).resolve()
            payload = arrays[key].tobytes(); path.write_bytes(payload)
            specs[name] = dict(path=path, offset=0, length=len(payload),
                               identity=identity(path.stat()), sha256=hashlib.sha256(payload).hexdigest())
        self.guard.enter()
        try:
            self.arena = Arena(specs, admission_check=lambda remaining: None, max_bytes=2**20)
            self.guard.track(self.arena); H.dontfork_arena(self.arena)
            self.arena.load()
            H.assert_dontfork(self.arena)
            self.graph = S.Graph(self.arena, metadata)
            self.native = S.Native(self.graph)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.native is not None:
            self.native.close()
        if self.graph is not None:
            self.graph.close()
        if self.arena is not None:
            self.arena.close()
            self.guard.confirm_released(self.arena)
        if self.guard.active:
            self.guard.release()
        self.temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
