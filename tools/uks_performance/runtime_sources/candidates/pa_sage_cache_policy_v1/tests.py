"""Bounded CPU checks for rankings, alias mapping, serving counts and integration."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .common import ROOT, read, require
from .protocol import compile_protocol, validate, arms
from .selection import topk, degree_scores, reverse_pagerank, FrequencyProfile
from .mapping import logical_slots, storage_slots, primary_rows
from .oracle import FeatureOracle
from .metrics import serving, from_native_report
from .native_adapter import install, check_backend, required_capabilities


def simple_cache(gpu_pages=1, hot=(0,)):
    features = np.arange(4 * 128, dtype=np.float32).reshape(4, 128)
    storage = np.full(24, -1, dtype=np.int64)
    storage[[0, 8]] = 0
    storage[[1, 9]] = 1
    storage[16] = 2
    storage[17] = 3
    payload = np.zeros((24, 128), dtype=np.float32)
    valid = storage >= 0
    payload[valid] = features[storage[valid]]
    return FeatureOracle(storage, payload, features, np.array(hot, dtype=np.int64), gpu_pages * 4096), features


class Tests(unittest.TestCase):
    def test_topk_ties_boundaries_and_invalid_scores(self):
        scores = np.array([7, 7, 9, 0, 7], dtype=np.int64)
        self.assertEqual(topk(scores, 3).tolist(), [0, 1, 2])
        self.assertEqual(topk(scores, 0).tolist(), [])
        self.assertEqual(topk(scores, 5).tolist(), list(range(5)))
        self.assertEqual(topk(np.zeros(4), 2).tolist(), [0, 1])
        for bad in (np.array([np.nan]), np.array([np.inf]), np.array([-1])):
            with self.assertRaises(ValueError):
                topk(bad, 1)
        with self.assertRaises(ValueError):
            topk(scores, 6)

    def test_reverse_pagerank_direction_parallel_edges_and_dangling(self):
        ptr = np.array([0, 0, 2, 4, 4], dtype=np.int64)
        idx = np.array([0, 0, 1, 2], dtype=np.int64)
        self.assertEqual(degree_scores(ptr, idx).tolist(), [0, 2, 2, 0])
        adjacency = np.zeros((4, 4))
        for src, dst in [(0, 1), (0, 1), (1, 2), (2, 2)]:
            adjacency[src, dst] += 1
        reversed_adjacency = adjacency.T
        transition = np.zeros((4, 4))
        for i, row in enumerate(reversed_adjacency):
            transition[i] = row / row.sum() if row.sum() else np.full(4, .25)
        expected = np.full(4, .25)
        for _ in range(20):
            expected = .15 / 4 + .85 * transition.T.dot(expected)
        for chunk in (1, 3, 100):
            actual = reverse_pagerank(ptr, idx, chunk_edges=chunk)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        self.assertGreater(actual[0], actual[1])
        empty = reverse_pagerank(np.zeros(5, dtype=np.int64), np.array([], dtype=np.int64))
        np.testing.assert_array_equal(empty, np.full(4, .25))

    def test_frequency_counts_feature_requests_and_freezes_before_measurement(self):
        profile = FrequencyProfile(4, 23)
        profile.observe(np.array([0, 1], dtype=np.int64))
        profile.observe(np.array([1, 2], dtype=np.int64))
        self.assertEqual(profile.counts.tolist(), [1, 2, 1, 0])
        with self.assertRaises(ValueError):
            profile.observe(np.array([0, 0], dtype=np.int64))
        with self.assertRaises(ValueError):
            profile.observe(np.array([4], dtype=np.int64))
        result = profile.freeze()
        self.assertEqual(result['total_logical_requests'], 4)
        with self.assertRaises(ValueError):
            profile.observe(np.array([1], dtype=np.int64))
        with self.assertRaises(ValueError):
            profile.counts[0] = 100
        with self.assertRaises(ValueError):
            FrequencyProfile(4, 23, purpose='measured_epoch')

    def test_exact_cpu_slots_keep_replicas_but_exclude_padding_and_page_neighbours(self):
        hot = np.array([0, 2], dtype=np.int64)
        slots = logical_slots(hot, 4)
        storage = np.array([0, 1, -1, 2, 0, 3, 2, -1], dtype=np.int64)
        self.assertEqual(storage_slots(storage, slots).tolist(), [1, 0, 0, 2, 1, 0, 2, 0])
        node_rows = np.array([0, 1, 3, 5], dtype=np.int64)
        self.assertEqual(primary_rows(hot, node_rows, storage, 'logical_node_real').tolist(), [0, 3])
        with self.assertRaises(ValueError):
            primary_rows(hot, node_rows, storage, 'physical_row_proxy')
        with self.assertRaises(ValueError):
            logical_slots(np.array([2, 0], dtype=np.int64), 4)

    def test_cpu_gpu_ssd_are_exclusive_and_features_remain_exact(self):
        cache, features = simple_cache()
        ids = np.array([0, 1, 0, 0, 2, 1], dtype=np.int64)
        rows = np.array([0, 1, 8, 0, 16, 1], dtype=np.int64)
        np.testing.assert_array_equal(cache.fetch(ids, rows), features[ids])
        r = cache.report()
        self.assertEqual((r['cpu_served_rows'], r['gpu_hit_rows'], r['ssd_served_rows']), (2, 1, 3))
        self.assertEqual(r['combined_hit_ratio'], .5)
        with self.assertRaises(ValueError):
            serving(6, 3, 1, 3, 4)

    def test_fifo_does_not_become_lru_and_static_does_not_retain_gpu_pages(self):
        cache, _ = simple_cache(gpu_pages=2, hot=())
        cache.fetch(np.array([1, 1, 1, 2], dtype=np.int64), np.array([1, 9, 1, 16], dtype=np.int64))
        self.assertEqual(list(cache.pages), [1, 2])
        static, _ = simple_cache(gpu_pages=0)
        static.fetch(np.array([1, 1, 0], dtype=np.int64), np.array([1, 1, 8], dtype=np.int64))
        self.assertEqual(static.report()['gpu_hit_rows'], 0)
        self.assertEqual(static.report()['ssd_served_rows'], 2)
        self.assertFalse(static.pages)

    def test_corrupt_replica_or_request_mapping_rejected(self):
        cache, features = simple_cache()
        payload = cache.payload.copy()
        payload[8, 0] += 1
        with self.assertRaises(ValueError):
            FeatureOracle(cache.storage, payload, features, np.array([0], dtype=np.int64), 4096)
        with self.assertRaises(ValueError):
            cache.fetch(np.array([0], dtype=np.int64), np.array([1], dtype=np.int64))

    def test_native_adapter_checks_capabilities_before_any_allocation(self):
        class RecordingBackend:
            def __init__(self, capabilities):
                self.flags, self.calls, self.slots = capabilities, [], {}
            def capabilities(self):
                return self.flags
            def begin_exact_cpu_cache(self, rows, extent, row_bytes):
                self.calls.append(('begin', rows.tolist(), extent, row_bytes))
            def write_cpu_row_map(self, lo, rows):
                self.slots.update({lo + i: int(v) for i, v in enumerate(rows)})
            def finish_exact_cpu_cache(self):
                self.calls.append(('finish',))
            def configure_gpu_cache(self, policy, capacity):
                self.calls.append((policy, capacity))
        policies = arms(2, 4096)
        storage = np.array([0, 1, -1, 2, 0, 3, 2, -1], dtype=np.int64)
        primary = np.array([0, 1, 3, 5], dtype=np.int64)
        hot = np.array([0, 2], dtype=np.int64)
        for policy in policies.values():
            b = RecordingBackend({key: True for key in required_capabilities(policy)})
            receipt = install(b, policy, hot, primary, storage, chunk_rows=3)
            self.assertEqual(list(b.slots.values()), [1, 0, 0, 2, 1, 0, 2, 0])
            self.assertEqual(receipt['row_map_gpu_bytes'], 32)
            self.assertEqual(b.calls[-1], (policy['gpu_policy'], policy['gpu_feature_cache_bytes']))
        b = RecordingBackend({})
        with self.assertRaises(ValueError):
            install(b, policies['degree'], hot, primary, storage)
        self.assertFalse(b.calls)

    def test_paper_protocol_does_not_hide_capacity_or_reuse_wrong_scores(self):
        p = compile_protocol()
        self.assertEqual(p['arms']['freq']['cpu_feature_bytes'], p['arms']['digit']['cpu_feature_bytes'])
        self.assertEqual(p['arms']['freq']['gpu_feature_cache_bytes'], 0)
        self.assertEqual(p['arms']['digit']['gpu_feature_cache_bytes'], 4 * 2**30)
        self.assertFalse(p['implementation']['native_ready'])
        for key, value in [('epochs', 20), ('evaluation', 'enabled')]:
            q = copy.deepcopy(p)
            q[key] = value
            with self.assertRaises(ValueError):
                validate(q)
        q = copy.deepcopy(p)
        q['arms']['degree']['gpu_policy'] = 'legacy'
        with self.assertRaises(ValueError):
            validate(q)

    def test_existing_native_report_denominators_and_corruption(self):
        path = ROOT / 'results/pa_sage_layout_shared_resume_20260925_v3/calibration/real_first/full_digit_full_accepted.json'
        report = read(path)
        m = from_native_report(report)
        self.assertEqual(m['logical_requests'], 291925962)
        self.assertEqual(m['cpu_served_rows'], 179446839)
        self.assertEqual(m['gpu_hit_rows'], 49780222)
        self.assertAlmostEqual(m['cpu_hit_ratio'] + m['gpu_hit_ratio'] + m['ssd_request_fraction'], 1.)
        bad = copy.deepcopy(report)
        bad['epochs'][0]['training']['feature']['cpu'] += 1
        with self.assertRaises(ValueError):
            from_native_report(bad)


def run():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    require(result.wasSuccessful(), 'CPU regression failed')
    return dict(passed=True, tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                native_execution=False, raw_ssd_access=False)
