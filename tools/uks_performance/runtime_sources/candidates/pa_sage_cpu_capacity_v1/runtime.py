"""Deferred GIDS connection. Importing this module performs no CUDA/SSD work."""
import importlib
import sys
from pathlib import Path
from .backend import allocation, install_on_loader
from .common import HERE, ROOT, GRID, BINARY, read, require, require_grid_complete, binary_receipt
from .counters import snapshot, interval


def loader_kwargs(protocol, arm, manifest, geometry):
    require(arm in protocol['arms'], 'Unknown cache policy')
    require(protocol['features']['mode'] == 'logical_node_real', 'Replica aliasing requires real features')
    require(protocol['layout']['point']['id'] == 'g2_r20' and
            manifest['grouping']['group_size'] == 2, 'Expected the fixed g2/r20 layout')
    require(manifest['feature']['row_bytes'] == 512 and manifest['feature']['dim'] == 128 and
            manifest['io']['page_size'] == 4096, 'Unsupported feature/page geometry')
    offset = protocol['features']['pool_offset']
    require(type(offset) is int and offset >= 0 and offset % 4096 == 0, 'Unaligned real payload')
    size = manifest['feature']['num_storage_rows'] * 512
    require(size <= protocol['features']['verified_bytes'], 'Payload exceeds the verified real extent')
    capacity = allocation(protocol['arms'][arm])
    return dict(page_size=4096, off=offset, cache_dim=128,
        num_ele=manifest['feature']['num_storage_rows'] * 128, num_ssd=1, ssd_list=[0],
        cache_size=capacity['gpu_dma_allocation_bytes'] // 2**20, ctrl_idx=0,
        window_buffer=False, accumulator_flag=False, feature_index_mode='explicit',
        gpu_cache_policy='fifo', cpu_feature_path='mapped', mixed_io=True,
        device_io_stats=True,
        mixed_io_geometry=geometry.with_payload_offset(offset).to_native_mapping())


def prepare_imports():
    """Used only in a future fresh, admitted native worker holding the SSD lock."""
    require_grid_complete(read(GRID / 'status.json'))
    binary_receipt()
    require(not any(name == 'GIDS' or name.startswith('GIDS.') or
                    name == 'BAM_Feature_Store' or name.startswith('BAM_Feature_Store.')
                    for name in sys.modules), 'Policy worker must start in a fresh process')
    from candidates.pa_sage_bidir_native_v2.common import setup
    setup()
    sys.path.insert(0, str(HERE / 'runtime'))
    module = importlib.import_module('BAM_Feature_Store.BAM_Feature_Store')
    require(Path(module.__file__).resolve() == BINARY.resolve() and module.cache_policy_abi() == 2,
            'Wrong native library')
    return importlib.import_module('GIDS').GIDS, module


def create_loader(protocol, arm, manifest, geometry, hot, primary, storage):
    """Caller supplies validated inputs, live admission and an exclusive SSD lock.

    Not a launcher: graph setup, monitor, short acceptance and process lifecycle
    must be provided by the later isolated worker, before this function is used.
    """
    kwargs = loader_kwargs(protocol, arm, manifest, geometry)
    cls, module = prepare_imports()
    loader = cls(**kwargs)
    receipt = install_on_loader(loader, module, protocol['arms'][arm], hot, primary, storage)
    return loader, receipt


def begin_epoch(loader):
    loader.BAM_FS.begin_useful_io_region()
    return snapshot(loader.BAM_FS)


def finish_window(loader, before, logical_rows, complete_epoch=False):
    after = snapshot(loader.BAM_FS)
    return after, interval(before, after, logical_rows, complete_region=complete_epoch)
