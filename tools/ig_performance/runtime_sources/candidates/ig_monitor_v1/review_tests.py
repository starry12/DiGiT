"""CPU-only adversarial review checks: no CUDA, NVML, GPU or SSD access."""
import copy
import json
import unittest

from candidates.ig_monitor_v1.review import BACKEND, MEMORY_API, MEMORY_DEFINITION, assess_monitor


def fixture():
    records = []
    for index in range(1, 15):
        at = 10.0 + (index - 1) * 0.5
        records.append(dict(time_unix=at, query_started_unix=at - 0.01,
                            query_finished_unix=at, device_used_bytes=1024 + index,
                            device_total_bytes=8192, device_free_bytes=8192 - 1024 - index - 512,
                            device_reserved_bytes=512, memory_api=MEMORY_API,
                            memory_used_definition=MEMORY_DEFINITION,
                            utilization_percent=30, monitor_rss_bytes=20 * 2**20,
                            supervisor_rss_bytes=9 * 2**20, sampler_rss_bytes=11 * 2**20,
                            monitor_pid=200, monitor_parent_pid=100, sampler_pid=201, sampler_parent_pid=200,
                            physical_gpu_uuid='GPU-fixture', physical_gpu_index=2,
                            backend=BACKEND, sequence=index))
    summary = dict(mode='external_small_process', pid=200, parent_pid=100, gpu='2',
                   samples=len(records), errors=[], peak_rss_bytes=22 * 2**20,
                   supervisor_peak_rss_bytes=10 * 2**20, sampler_peak_rss_bytes=12 * 2**20,
                   peak_device_used_bytes=max(x['device_used_bytes'] for x in records),
                   complete=True, passed=True, sampler_stopped=True, query_timeout_seconds=5.0,
                   sampler_shutdown_action='terminated', sampler_returncode=-15,
                   backend=BACKEND, physical_gpu_uuid='GPU-fixture',
                   physical_gpu_index=2, sampler_pid=201, started_unix=9.0, finished_unix=16.6)
    ready = dict(pid=200, parent_pid=100, passed=True, first_sample=copy.deepcopy(records[0]),
                 backend=BACKEND, sampler_pid=201, physical_gpu_uuid='GPU-fixture', physical_gpu_index=2)
    worker = dict(pid=300, started_unix=10.1, finished_unix=17.0)
    resources = dict(worker_pid=300, mode='checkpoints_only_external_sampler',
                     background_monitor_in_worker=False, monitor_samples=0, monitor_error=None,
                     checkpoints=[dict(stage='start', time_unix=10.2, device_used_bytes=900),
                                  dict(stage='training_complete', time_unix=16.4, device_used_bytes=1500)],
                     observed_peak_device_used_bytes=1500)
    return dict(summary=summary, records=records, ready=ready, worker=worker,
                controller_pid=100, resources=resources, returncode=0, gpu=2)


class MonitorReviewTests(unittest.TestCase):
    def rejected(self, value, text):
        with self.assertRaisesRegex(RuntimeError, text):
            assess_monitor(**value)

    def test_success_preserves_checkpoint_and_sample_peaks(self):
        result = assess_monitor(**fixture())
        self.assertIs(result['strict_monitor_passed'], True)
        self.assertEqual(result['sample_count'], 13)
        self.assertEqual(result['observed_peak_device_used_bytes'], 1500)
        self.assertEqual(result['monitor_peak_rss_bytes'], 22 * 2**20)
        self.assertEqual(result['timeout_count'], 0)
        self.assertEqual(result['review_policy']['max_error_fraction'], 0.0)

    def test_any_error_rejects_even_sparse_and_outside_training(self):
        value = fixture()
        error = dict(time_unix=9.5, error='NVML query watchdog timed out', error_kind='query_timeout')
        value['records'].insert(0, error)
        value['summary'].update(errors=[error], passed=False)
        self.rejected(value, 'zero errors')

    def test_forged_passed_or_removed_error_fails_reconciliation(self):
        value = fixture()
        error = dict(time_unix=9.5, error='NVML_ERROR_UNKNOWN')
        value['records'].insert(0, error)
        self.rejected(value, 'summary differs')
        value['summary']['errors'] = [error]
        self.rejected(value, 'original monitor acceptance')

    def test_zero_error_exact_fifteen_second_gap_rejected(self):
        value = fixture()
        for sample in value['records'][7:]:
            for field in ('time_unix', 'query_started_unix', 'query_finished_unix'):
                sample[field] += 14.5
        value['summary']['finished_unix'] += 14.5
        value['worker']['finished_unix'] += 14.5
        value['resources']['checkpoints'][-1]['time_unix'] += 14.5
        self.rejected(value, 'Excessive monitoring gap')

    def test_wrong_sample_gpu_or_process_identity_rejected(self):
        for field, changed in [('physical_gpu_index', 3), ('physical_gpu_uuid', 'GPU-other'),
                               ('sampler_pid', 301), ('sampler_parent_pid', 101), ('monitor_parent_pid', 101),
                               ('backend', 'nvidia-smi')]:
            with self.subTest(field=field):
                value = fixture()
                value['records'][5][field] = changed
                self.rejected(value, 'sample process/device identity')

    def test_wrong_ownership_and_nonindependent_sampler_rejected(self):
        value = fixture()
        value['ready']['parent_pid'] = 101
        self.rejected(value, 'ownership')
        value = fixture()
        value['summary']['sampler_pid'] = value['worker']['pid']
        self.rejected(value, 'independent')

    def test_combined_rss_budget_and_peak_reconciliation(self):
        value = fixture()
        value['summary']['peak_rss_bytes'] = 70 * 2**20
        value['summary']['sampler_peak_rss_bytes'] = 60 * 2**20
        self.rejected(value, 'RSS exceeds')
        value = fixture()
        value['records'][5]['monitor_rss_bytes'] += 1
        self.rejected(value, 'RSS accounting')

    def test_summary_and_readiness_tampering(self):
        for target, key, changed, pattern in [
                ('summary', 'samples', 13, 'summary differs'),
                ('summary', 'peak_device_used_bytes', 0, 'peak differs')]:
            with self.subTest(key=key):
                value = fixture()
                value[target][key] = changed
                self.rejected(value, pattern)
        value = fixture()
        value['ready']['first_sample']['device_used_bytes'] = 0
        self.rejected(value, 'Readiness sample differs')

    def test_sample_gap_cannot_hide_in_query_or_clock(self):
        value = fixture()
        value['records'][4]['query_started_unix'] -= 6
        self.rejected(value, 'query timing')
        value = fixture()
        value['records'][4]['time_unix'] = float('nan')
        self.rejected(value, 'invalid samples')
        value = fixture()
        value['records'][4]['query_started_unix'] -= .6
        self.rejected(value, 'queries')

    def test_training_endpoint_and_checkpoint_coverage(self):
        value = fixture()
        value['worker']['finished_unix'] = 21.0
        value['resources']['checkpoints'][-1]['time_unix'] = 20.0
        self.rejected(value, 'cover training lifetime')
        value = fixture()
        value['resources']['checkpoints'][-1]['stage'] = 'not_complete'
        self.rejected(value, 'Incomplete worker checkpoints')
        value = fixture()
        value['resources']['checkpoints'][-1]['time_unix'] = 18.0
        self.rejected(value, 'outside worker lifetime')

    def test_too_few_worker_samples(self):
        value = fixture()
        value['worker']['started_unix'] = 12.1
        value['resources']['checkpoints'][0]['time_unix'] = 12.2
        self.rejected(value, 'Insufficient worker samples')

    def test_missing_samples_rejected_even_with_matching_summary(self):
        value = fixture()
        del value['records'][5]
        value['summary']['samples'] -= 1
        self.rejected(value, 'sample sequence')

    def test_unexpected_sampler_shutdown_cannot_be_reported_passed(self):
        for action, code in [('already_exited', 0), ('already_exited', 1), ('terminated', 0)]:
            with self.subTest(action=action, code=code):
                value = fixture()
                value['summary'].update(sampler_shutdown_action=action, sampler_returncode=code)
                self.rejected(value, 'bounded intentional shutdown')
        value = fixture()
        value['summary']['sampler_stopped'] = False
        self.rejected(value, 'finish normally')

    def test_query_timeout_policy_cannot_be_widened(self):
        for timeout in (5.1, float('inf'), 0.0):
            with self.subTest(timeout=timeout):
                value = fixture()
                value['summary']['query_timeout_seconds'] = timeout
                self.rejected(value, 'timeout exceeds policy')

    def test_memory_api_and_used_definition_cannot_change(self):
        for field, value in [('memory_api', 'nvmlDeviceGetMemoryInfo'),
                             ('memory_used_definition', 'includes reserved')]:
            with self.subTest(field=field):
                data = fixture()
                data['records'][5][field] = value
                self.rejected(data, 'memory API or used-memory definition')

    def test_memory_counters_are_nonnegative_integer_bytes(self):
        for field in ('device_total_bytes', 'device_used_bytes', 'device_free_bytes',
                      'device_reserved_bytes'):
            for value in (-1, 1.5, True):
                with self.subTest(field=field, value=value):
                    data = fixture()
                    data['records'][5][field] = value
                    self.rejected(data, 'memory byte counters or conservation')

    def test_memory_conservation_cannot_hide_reserved_bytes(self):
        data = fixture()
        data['records'][5]['device_used_bytes'] += data['records'][5]['device_reserved_bytes']
        self.rejected(data, 'memory byte counters or conservation')

    def test_nonzero_monitor_exit_rejected(self):
        value = fixture()
        value['returncode'] = 1
        self.rejected(value, 'finish normally')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(MonitorReviewTests))
    print(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun,
                          failures=len(result.failures), errors=len(result.errors))))
    raise SystemExit(0 if result.wasSuccessful() else 1)
