"""Concrete exact-row installer for the new ABI; no device work at import time."""
from pathlib import Path
import numpy as np
from candidates.uks_native_v1.mapping import install
from .common import SCRATCH_BYTES, BINARY, require, binary_receipt

CAPABILITIES = dict(exact_logical_cpu_rows=True, replica_aliases=True,
    padding_uncached=True, useful_io_counters=True, exclusive_serving_counters=True,
    static_gpu_bypass=True, fifo_gpu_first=True, legacy_gpu_first=True)


def allocation(arm):
    policy, size = arm['gpu_policy'], arm['gpu_feature_cache_bytes']
    require(policy in ('bypass', 'fifo', 'legacy'), 'Unsupported policy')
    require(type(size) is int and size >= 0 and size % 2**20 == 0, 'Invalid capacity')
    require(size == 0 if policy == 'bypass' else size >= SCRATCH_BYTES, 'Wrong feature-cache budget')
    return dict(gpu_feature_cache_bytes=size,
        gpu_dma_allocation_bytes=SCRATCH_BYTES if policy == 'bypass' else size,
        gpu_preload_scratch_bytes=SCRATCH_BYTES,
        additional_gpu_scratch_bytes=SCRATCH_BYTES+(SCRATCH_BYTES if policy == 'bypass' else 0),
        pair_dma_staging_bytes=SCRATCH_BYTES,pair_hash_bytes='power-of-two capacity at least 2*batch input rows, int32 entries, DiGiT only',
        scratch_note='FIFO reuses its still-cold allocation during CPU preload',
        persistent_gpu_cache=policy != 'bypass', transport_allocator_policy='fifo' if policy=='bypass' else policy)


class ExactInstaller:
    """Stateful Python checks shared by real native use and bounded CPU fakes.

    Construct this class only through attach() in device code. CPU tests use a
    recording implementation; its presence is not proof of native acceptance.
    """
    def __init__(self, fs):
        self.fs = fs
        self.state = 'fresh'
        self.cursor = self.extent = self.rows = 0

    def capabilities(self):
        return dict(CAPABILITIES)

    def begin_exact_cpu_cache(self, rows, extent, row_bytes):
        require(self.state == 'fresh', 'CPU cache already initialized')
        require(type(extent) is int and extent > 0 and extent % 4 == 0 and row_bytes == 1024,
                'Invalid payload extent/row width')
        require(isinstance(rows, np.ndarray) and rows.dtype == np.int64 and rows.ndim == 1 and
                rows.flags.c_contiguous and 0 <= len(rows) < 2**31, 'Invalid primary-row vector')
        require(not len(rows) or (rows.min() >= 0 and rows.max() < extent), 'Primary address outside payload')
        self.fs.policy_begin_cpu_cache(rows, extent)
        self.rows, self.extent, self.state = len(rows), extent, 'installing'

    def write_cpu_row_map(self, lo, slots):
        require(self.state == 'installing' and type(lo) is int and lo == self.cursor,
                'Duplicate, missing or out-of-order map chunk')
        require(isinstance(slots, np.ndarray) and slots.dtype == np.uint32 and slots.ndim == 1 and
                slots.flags.c_contiguous and 0 < len(slots) <= self.extent - lo,
                'Wrong slot buffer/extent')
        require(slots.max() <= self.rows, 'CPU slot exceeds hot-set capacity')
        self.fs.policy_write_cpu_map(lo, slots)
        self.cursor += len(slots)

    def finish_exact_cpu_cache(self):
        require(self.state == 'installing' and self.cursor == self.extent, 'Incomplete CPU row map')
        self.fs.policy_finish_cpu_cache()
        self.state = 'installed'

    def configure_gpu_cache(self, policy, capacity):
        require(self.state == 'installed', 'Cannot change policy during a run')
        allocation(dict(gpu_policy=policy, gpu_feature_cache_bytes=capacity))
        self.fs.policy_configure({'bypass':1,'fifo':2,'legacy':3}[policy], capacity)
        self.state = 'configured'


def attach(loader, native_module):
    receipt = binary_receipt()
    require(Path(native_module.__file__).resolve() == BINARY.resolve(), 'Loaded a different experiment binary')
    require(hasattr(native_module, 'cache_policy_abi') and native_module.cache_policy_abi() == 4 and
            native_module.cache_policy_row_bytes() == 1024 and native_module.cache_policy_page_bytes() == 1024,
            'Backend does not implement the isolated policy ABI')
    fs = loader.BAM_FS
    for name in ('policy_begin_cpu_cache', 'policy_write_cpu_map', 'policy_finish_cpu_cache',
                 'policy_configure', 'policy_stats', 'begin_useful_io_region'):
        require(callable(getattr(fs, name, None)), 'Native method missing: ' + name)
    return ExactInstaller(fs)


def install_on_loader(loader, native_module, arm, hot, primary, storage):
    """All graph/layout inputs must be bound by the future native worker first."""
    require(loader.cpu_feature_path == 'mapped', 'Exact rows need mapped CPU reads')
    result = install(attach(loader, native_module), arm, hot, primary, storage,
                     feature_mode='logical_synthetic')
    result['allocation'] = allocation(arm)
    # Native preload uses direct I/O, leaving resident tags and request counts cold.
    require(loader.get_gpu_cache_stats()['resident_pages'] == 0, 'CPU preload warmed GPU cache')
    require(sum(loader.get_feature_access_stats().values()) == 0, 'Preload counted as training requests')
    loader.reset_device_io_stats()
    return result
