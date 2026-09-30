"""Small CPU-only tests for the new ABI adapter, counters and deferred build gate."""
import copy
import io
import unittest
from unittest.mock import patch
import numpy as np
from candidates.pa_sage_cache_policy_v1.native_adapter import install
from candidates.pa_sage_cache_policy_v1.protocol import arms
from .backend import ExactInstaller, allocation
from .common import require_grid_complete, SCRATCH_BYTES
from .counters import POLICY, GPU, DEVICE, snapshot, interval
from .acceptance import accept_arm, compare, TRACE_KEYS


class FakeStore:
    """Records ABI calls only. Never claims to model CUDA scheduling or NVMe."""
    def __init__(self):
        self.calls = []
        self.slots = []
    def policy_begin_cpu_cache(self, rows, extent):
        self.calls.append(('begin', rows.tolist(), extent))
    def policy_write_cpu_map(self, lo, slots):
        self.calls.append(('map', lo, len(slots)))
        self.slots += slots.tolist()
    def policy_finish_cpu_cache(self):
        self.calls.append(('finish',))
    def policy_configure(self, mode, capacity):
        self.calls.append(('configure', mode, capacity))


def counters(mode=1, end=False):
    # Six logical requests: 2 CPU, 4 on the GPU/SSD route. FIFO: 1 hit, 3 fills.
    requests = 4 if end else 0
    hits = int(end and mode == 2)
    fills = requests - hits
    p = dict(zip(POLICY, [1, mode, 3, 24, 2,
        0 if mode == 1 else SCRATCH_BYTES, SCRATCH_BYTES, 2,
        requests if mode == 1 else 0, 0]))
    g = dict(zip(GPU, [1, 4096, requests if mode == 2 else 0, hits,
        fills if mode == 2 else 0, 0, fills if mode == 2 else 0,
        fills if mode == 2 else 0, 0, fills if mode == 2 else 0]))
    # One 512-B replay per primary command in this synthetic fixture only.
    d = dict(zip(DEVICE, [1, 0, fills * 2, fills * 2, fills * 4608,
        fills * 100, fills * 200, 100 if end else 0, 1 if end else 0, fills, fills * 512]))
    class Source:
        def policy_stats(self): return list(p.values())
        def get_feature_access_stats(self): return [2 if end else 0, requests]
        def get_gpu_cache_stats(self): return list(g.values())
        def get_device_io_stats(self): return list(d.values())
        def get_useful_io_stats(self): return [1, 1, fills * 4096, fills * 512, requests * 512, requests]
    return snapshot(Source())


def report(arm):
    mode = 2 if arm == 'digit' else 1
    region = interval(counters(mode), counters(mode, True), 6, complete_region=True)
    result = dict(kind='native_cache_policy_full_epoch', native=True, source_only=False,
        worker_returncode=0, arm=arm, source_sha256='a'*64, protocol_sha256='b'*64,
        native_short_receipt_sha256='c'*64, epochs=1, updates=1179,
        evaluation='disabled', finite_loss_and_gradients=True, training_seconds=1.,
        monitor=dict(passed=True, backend='nvidia-smi'), region=region,
        windows=[dict(updates=1179, region=copy.deepcopy(region))], cpu_rows=2,
        hot_nodes_sha256='d'*64)
    result.update({k: 'e'*64 for k in TRACE_KEYS})
    return result


class Tests(unittest.TestCase):
    def test_four_arm_exact_installer_aliases_and_padding(self):
        storage = np.array([0, 1, -1, 2, 0, 3, 2, -1], dtype=np.int64)
        hot = np.array([0, 2], dtype=np.int64)
        primary = np.array([0, 1, 3, 5], dtype=np.int64)
        for name, arm in arms(2, SCRATCH_BYTES).items():
            fs = FakeStore()
            backend = ExactInstaller(fs)
            install(backend, arm, hot, primary, storage, chunk_rows=3)
            self.assertEqual(fs.slots, [1, 0, 0, 2, 1, 0, 2, 0])
            self.assertEqual(fs.calls[0], ('begin', [0, 3], 8))
            self.assertEqual(fs.calls[-1], ('configure', 2 if name == 'digit' else 1,
                                          SCRATCH_BYTES if name == 'digit' else 0))
            with self.assertRaises(ValueError):
                backend.configure_gpu_cache('bypass', 0)

    def test_partial_repeated_or_invalid_map_cannot_finish(self):
        backend = ExactInstaller(FakeStore())
        backend.begin_exact_cpu_cache(np.array([0, 3], np.int64), 8, 512)
        with self.assertRaises(ValueError): backend.finish_exact_cpu_cache()
        for lo, slots in [(1, np.zeros(2, np.uint32)), (0, np.array([3], np.uint32)),
                          (0, np.zeros(9, np.uint32)), (0, np.zeros(2, np.int64))]:
            with self.assertRaises(ValueError): backend.write_cpu_row_map(lo, slots)
        self.assertEqual(backend.cursor, 0)
        backend.write_cpu_row_map(0, np.zeros(4, np.uint32))
        with self.assertRaises(ValueError): backend.write_cpu_row_map(0, np.zeros(4, np.uint32))

    def test_invalid_installation_fails_before_native_call(self):
        fs = FakeStore()
        for rows, extent in [(np.array([-1], np.int64), 8), (np.array([8], np.int64), 8),
                             (np.array([0], np.int64), 7), (np.array([0], np.int32), 8)]:
            with self.assertRaises(ValueError): ExactInstaller(fs).begin_exact_cpu_cache(rows, extent, 512)
        self.assertFalse(fs.calls)

    def test_feature_capacity_distinct_from_dma_scratch(self):
        for name, arm in arms(2, 4*2**30).items():
            a = allocation(arm)
            self.assertEqual(a['gpu_feature_cache_bytes'], 4*2**30 if name == 'digit' else 0)
            self.assertEqual(a['gpu_dma_allocation_bytes'], 4*2**30 if name == 'digit' else SCRATCH_BYTES)
        for arm in [dict(gpu_policy='legacy', gpu_feature_cache_bytes=0),
                    dict(gpu_policy='bypass', gpu_feature_cache_bytes=SCRATCH_BYTES),
                    dict(gpu_policy='fifo', gpu_feature_cache_bytes=0)]:
            with self.assertRaises(ValueError): allocation(arm)

    def test_actual_loader_settings_preserve_addresses_and_separate_capacities(self):
        from .runtime import loader_kwargs
        p = dict(arms=arms(2, 4*2**30), layout=dict(point=dict(id='g2_r20')),
                 features=dict(mode='logical_node_real', pool_offset=3*2**40, verified_bytes=24*512))
        m = dict(grouping=dict(group_size=2), feature=dict(row_bytes=512, dim=128, num_storage_rows=24),
                 io=dict(page_size=4096))
        class Geometry:
            def with_payload_offset(self, offset): self.offset = offset; return self
            def to_native_mapping(self): return dict(offset=self.offset)
        expected = None
        for arm in p['arms']:
            kwargs = loader_kwargs(p, arm, m, Geometry())
            self.assertEqual(kwargs['cache_size'], 4096 if arm == 'digit' else 16)
            self.assertEqual(kwargs['num_ele'], 24*128)
            kwargs.pop('cache_size')
            if expected is not None: self.assertEqual(kwargs, expected)
            expected = kwargs
        for key, value in [('mode','physical_row_proxy'), ('pool_offset',1), ('verified_bytes',512)]:
            bad = copy.deepcopy(p); bad['features'][key] = value
            with self.assertRaises(ValueError): loader_kwargs(bad, 'freq', m, Geometry())

    def test_direct_and_fifo_partition_and_primary_replay_bytes(self):
        for mode in (1, 2):
            r = interval(counters(mode), counters(mode, True), 6, complete_region=True)
            self.assertEqual(r['serving']['cpu_served_rows'], 2)
            self.assertEqual(r['serving']['gpu_hit_rows'], 0 if mode == 1 else 1)
            self.assertEqual(r['serving']['ssd_served_rows'], 4 if mode == 1 else 3)
            self.assertEqual(r['device']['completed_bytes'], r['device']['primary_bytes'] + r['device']['replay_bytes'])

    def test_static_hidden_gpu_residency_rejected(self):
        before, after = counters(), counters(end=True)
        for key in ('requests', 'hits', 'resident_pages', 'inserts'):
            bad = copy.deepcopy(after); bad['gpu'][key] = 1
            with self.assertRaises(ValueError): interval(before, bad, 6)

    def test_missing_io_region_reset_and_bad_address_rejected(self):
        before, after = counters(), counters(end=True)
        for section, key, value in [('device', 'submitted_commands', 9), ('policy', 'address_errors', 1),
            ('policy', 'gpu_feature_bytes', 4096), ('useful_io', 'region_id', 2),
            ('device', 'enabled', False), ('feature', 'gpu_ssd', 3),
            ('useful_io', 'ssd_useful_bytes', 512), ('policy', 'bypass_ssd_rows', 0)]:
            bad = copy.deepcopy(after); bad[section][key] = value
            with self.assertRaises((ValueError, RuntimeError)): interval(before, bad, 6, True)

    def test_window_can_consume_previous_fill_but_whole_region_cannot(self):
        before, after = counters(2), counters(2, True)
        # Two previous fills, now one new fill and three consumed rows.
        before['useful_io']['ssd_fill_bytes'] = 8192
        before['useful_io']['ssd_useful_bytes'] = 0
        after['useful_io']['ssd_useful_bytes'] = 1536
        # Full region valid, and a useful interval remains separately reusable.
        from candidates.io_accounting_v1.accounting import useful_interval
        value = useful_interval(before['useful_io'], after['useful_io'], dict(gpu_ssd=4))
        self.assertEqual(value['ssd_fill_bytes'], 4096)
        # Malformed whole region (consumption outside its fills) must be rejected.
        bad = interval(counters(2), counters(2, True), 6, True)
        bad['useful_io']['ssd_useful_bytes'] = bad['useful_io']['ssd_fill_bytes'] + 512
        from candidates.io_accounting_v1.accounting import validate_region
        with self.assertRaises(RuntimeError): validate_region(bad)

    def test_grid_failure_missing_or_partial_completion_blocks_expensive_work(self):
        good = dict(complete=True, passed=True, completed=[str(i) for i in range(15)])
        require_grid_complete(good)
        for s in ({}, dict(good, complete=False), dict(good, passed=False),
                  dict(good, completed=['x']*15), dict(good, completed=['x'])):
            with self.assertRaises(ValueError): require_grid_complete(s)

    def test_build_blocks_before_subprocess_while_grid_active(self):
        from .build import execute
        with patch('candidates.pa_sage_cache_policy_v2.build.read', return_value={'complete': False}), \
             patch('subprocess.run') as run, patch('subprocess.check_output') as output:
            with self.assertRaises(ValueError): execute()
            run.assert_not_called(); output.assert_not_called()

    def test_acceptance_rejects_cpu_preflight_incomplete_or_wrong_monitor(self):
        good = report('degree')
        def accept(r): return accept_arm(r, 'degree', 'a'*64, 'b'*64, 'c'*64, arms(2, SCRATCH_BYTES)['degree'])
        accept(good)
        for key, val in [('native', False), ('kind', 'cpu_fixture'), ('worker_returncode', 1),
                         ('updates', 1178), ('native_short_receipt_sha256', ''),
                         ('training_seconds', float('nan')), ('cpu_rows', 3)]:
            bad = copy.deepcopy(good); bad[key] = val
            with self.assertRaises(ValueError): accept(bad)
        bad = copy.deepcopy(good); bad['monitor']['passed'] = False
        with self.assertRaises(ValueError): accept(bad)

    def test_comparison_requires_same_workload_and_freq_hotset(self):
        reports = {a: report(a) for a in ('degree', 'revpr', 'freq', 'digit')}
        self.assertFalse(compare(reports)['same_total_cache_capacity'])
        for name, key in [('digit', 'hot_nodes_sha256'), ('revpr', 'storage_trace_sha256')]:
            bad = copy.deepcopy(reports); bad[name][key] = 'f'*64
            with self.assertRaises(ValueError): compare(bad)


def run():
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(output.getvalue(), flush=True)
    if not result.wasSuccessful(): raise RuntimeError('New policy CPU checks failed')
    return dict(passed=True, tests=result.testsRun, evidence_kind='CPU mocks and synthetic counters only')
