"""CPU-only parity and lifecycle checks for the split bounded accumulator."""
from concurrent.futures import ThreadPoolExecutor
import random
import threading
import unittest

from candidates.pa_sage_512b_pair_v4.window import WindowAccumulator as SerialWindow
from .window import WindowAccumulator


class WindowTests(unittest.TestCase):
    def compare(self, sizes, threshold=6832.4, window_size=2, max_batches=4,
                max_rows=262144, cpu_mode='vary', split=True):
        source = [(range(size), i) for i, size in enumerate(sizes)]
        settings = dict(window_size=window_size, max_batches=max_batches,
                        max_rows=max_rows)
        old_events, new_events = [], []
        old_cpu, new_cpu = [], []
        tickets = []

        def result(batches):
            rows = sum(len(batch[0]) for batch in batches)
            fraction = {'none': 0, 'all': 1}.get(cpu_mode)
            if fraction is None:
                fraction = (0, .25, .9, 1)[batches[0][1] % 4]
            return [batch[1] for batch in batches], int(rows * fraction)

        def read(batches):
            old_events.append(('read', tuple(batch[1] for batch in batches)))
            features, cpu = result(batches)
            old_cpu.append(cpu)
            return features, cpu

        old = SerialWindow(source, read,
                           lambda batch: old_events.append(('hint', batch[1])),
                           threshold, **settings)
        old_output, old_estimates = [], []
        for batch, features in old:
            old_output.append((batch[1], features))
            old_estimates.append(old.previous_cpu_per_batch)

        def submit(batches):
            self.assertFalse(any(not ticket['collected'] for ticket in tickets))
            new_events.append(('read', tuple(batch[1] for batch in batches)))
            ticket = dict(result=result(batches), complete=False, collected=False)
            tickets.append(ticket)
            return ticket

        def collect(ticket):
            self.assertTrue(ticket['complete'], 'acquire must wait for modeled completion')
            self.assertFalse(ticket['collected'])
            ticket['collected'] = True
            new_cpu.append(ticket['result'][1])
            return ticket['result']

        pipe = WindowAccumulator(source, submit, collect,
                                 lambda batch: new_events.append(('hint', batch[1])),
                                 threshold, **settings)
        output, estimates = [], []
        if split:
            has_next = pipe.prepare_next()
            if has_next:
                tickets[-1]['complete'] = True
                current = pipe.acquire_next()
                while True:
                    # Exactly the intended training loop: prepare future work,
                    # compute current, then acquire future features.
                    has_next = pipe.prepare_next()
                    self.assertEqual(has_next, pipe.prepare_next())
                    output.append((current[0][1], current[1]))
                    estimates.append(pipe.previous_cpu_per_batch)
                    if pipe.has_outstanding_read:
                        pipe.outstanding_ticket['complete'] = True
                    if not has_next:
                        break
                    current = pipe.acquire_next()
        else:
            original_submit = pipe.submit_read
            def completed_submit(batches):
                ticket = original_submit(batches)
                ticket['complete'] = True
                return ticket
            pipe.submit_read = completed_submit
            for batch, features in pipe:
                output.append((batch[1], features))
                estimates.append(pipe.previous_cpu_per_batch)
        self.assertEqual(output, old_output)
        self.assertEqual(new_events, old_events)
        self.assertEqual(new_cpu, old_cpu)
        self.assertEqual(estimates, old_estimates)
        expected = old.summary()
        actual = pipe.summary()
        for key, value in expected.items():
            if key not in ('sample_ns', 'window_hint_ns'):
                self.assertEqual(actual[key], value, key)
        self.assertTrue(actual['drained'])
        self.assertEqual(actual['async_submit_calls'], old.read_calls)
        self.assertEqual(actual['async_collect_calls'], old.read_calls)
        self.assertEqual(actual['submitted_reads'], old.read_calls)
        self.assertEqual(actual['collected_reads'], old.read_calls)
        self.assertLessEqual(actual['max_outstanding_reads'], 1)
        self.assertEqual(actual['outstanding_reads'], 0)
        self.assertEqual(actual['prepared_batches'], 0)
        self.assertLessEqual(actual['max_pending_batches'], actual['retained_batch_limit'])
        self.assertLessEqual(actual['max_ready_features'], max_batches)
        self.assertFalse(pipe.prepare_next())
        with self.assertRaises(StopIteration):
            pipe.acquire_next()
        with self.assertRaises(StopIteration):
            next(pipe)
        return pipe, new_events

    def test_empty_short_tail_and_next_compatibility(self):
        for sizes in ([], [17], [10, 17], [10] * 3, [10] * 7 + [1]):
            for split in (False, True):
                with self.subTest(sizes=sizes, split=split):
                    self.compare(sizes, split=split)

    def test_complete_epoch_high_cpu_hit_and_tail(self):
        pipe, events = self.compare([100000] + [60000] * 1177 + [907], cpu_mode='all')
        self.assertEqual(pipe.delivered, 1179)
        self.assertEqual(sum(907 if i == 1178 else 1024
                             for kind, group in events if kind == 'read'
                             for i in group), 1207179)
        self.assertGreater(pipe.batch_cap_flushes, 0)
        self.assertLessEqual(pipe.max_read_rows, 262144)

    def test_row_cap_and_oversized_batches(self):
        pipe, _ = self.compare([100000] + [90000] * 23,
                               threshold=0, max_rows=200000, cpu_mode='all')
        self.assertGreater(pipe.row_cap_flushes, 0)
        self.assertLessEqual(pipe.max_read_rows, 200000)
        pipe, events = self.compare([300000, 10, 300001])
        self.assertEqual(pipe.oversized_single_batches, 2)
        self.assertIn(('read', (0,)), events)
        self.assertIn(('read', (2,)), events)

    def test_seeded_adversarial_trace_parity(self):
        rng = random.Random(0)
        for case in range(80):
            sizes = [rng.choice([0, 1, 100, 2000, 90000, 262144, 300000])
                     for _ in range(rng.randrange(1, 70))]
            with self.subTest(case=case):
                self.compare(sizes, threshold=rng.choice([0, 6832.4, 1e12]),
                             window_size=rng.choice([0, 1, 2, 7]),
                             max_batches=rng.choice([1, 2, 4]),
                             max_rows=rng.choice([200000, 262144]),
                             cpu_mode=rng.choice(['none', 'all', 'vary']))

    def test_prepare_is_idempotent_and_does_not_pop_ready(self):
        submitted, collected, hints = [], [], []
        def submit(batches):
            submitted.append([batch[1] for batch in batches])
            return list(batches)
        def collect(ticket):
            collected.append(ticket)
            return [batch[1] for batch in ticket], 0
        pipe = WindowAccumulator([(range(10), i) for i in range(8)],
                                 submit, collect, lambda batch: hints.append(batch[1]), 1e12)
        with self.assertRaisesRegex(RuntimeError, 'prepare_next'):
            pipe.acquire_next()
        self.assertTrue(pipe.prepare_next())
        self.assertEqual(len(collected), 0)
        first_state = (list(pipe.pending), list(pipe.ready), list(hints), list(submitted))
        self.assertTrue(pipe.prepare_next())
        self.assertEqual(first_state, (list(pipe.pending), list(pipe.ready), hints, submitted))
        self.assertEqual(pipe.acquire_next()[1], 0)
        with self.assertRaisesRegex(RuntimeError, 'prepare_next'):
            pipe.acquire_next()
        self.assertTrue(pipe.prepare_next())
        self.assertEqual(pipe.pending[0][1], 1)
        self.assertEqual(list(pipe.ready), [1, 2, 3])
        self.assertEqual(pipe.delivered, 1)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(len(collected), 1)
        self.assertEqual(pipe.acquire_next()[1], 1)
        self.assertEqual([features for _, features in pipe], list(range(2, 8)))
        self.assertTrue(pipe.summary()['drained'])

    def test_delayed_ticket_allows_work_between_submit_and_collect(self):
        start = threading.Event()
        finished = threading.Event()
        events = []
        def work(batches):
            if not start.wait(2):
                raise RuntimeError('Fixture was not released')
            finished.set()
            return [batch[1] for batch in batches], 0
        with ThreadPoolExecutor(max_workers=1) as executor:
            pipe = WindowAccumulator([(range(10), 0)],
                                     lambda batches: executor.submit(work, batches),
                                     lambda ticket: ticket.result(timeout=2),
                                     lambda batch: events.append('hint'), 0)
            try:
                self.assertTrue(pipe.prepare_next())
                self.assertFalse(finished.is_set())
                self.assertEqual(pipe.read_calls, 0)
                events.append('model_work')
            finally:
                start.set()
            self.assertEqual(pipe.acquire_next()[1], 0)
            self.assertTrue(finished.is_set())
            self.assertEqual(events, ['hint', 'model_work'])
            self.assertFalse(pipe.prepare_next())

    def test_none_is_valid_ticket_and_cpu_estimate_updates_only_at_collect(self):
        collected = []
        def collect(ticket):
            collected.append(ticket)
            return [0], 9
        pipe = WindowAccumulator([(range(10), 0)], lambda batches: None,
                                 collect, lambda batch: None, 0)
        self.assertTrue(pipe.prepare_next())
        self.assertTrue(pipe.has_outstanding_read)
        self.assertEqual(pipe.previous_cpu_per_batch, 0)
        self.assertEqual(pipe.acquire_next()[1], 0)
        self.assertEqual(collected, [None])
        self.assertEqual(pipe.previous_cpu_per_batch, 9)
        self.assertFalse(pipe.has_outstanding_read)

    def test_invalid_bounds(self):
        for settings in (dict(window_size=-1), dict(required_accesses=-1),
                         dict(max_batches=0), dict(max_rows=0)):
            args = dict(required_accesses=10)
            args.update(settings)
            with self.assertRaises(ValueError):
                WindowAccumulator([], lambda batches: None, lambda ticket: None,
                                  lambda batch: None, **args)

    def test_submit_failure_keeps_budget_and_prevents_retry(self):
        attempts = []
        def submit(batches):
            attempts.append(batches)
            raise MemoryError('fixture')
        pipe = WindowAccumulator(((range(60000), i) for i in range(100)), submit,
                                 lambda ticket: None, lambda batch: None, 1e12)
        with self.assertRaises(MemoryError):
            pipe.prepare_next()
        self.assertEqual(pipe.current_read, dict(batches=4, rows=240000, reason='batch_cap'))
        with self.assertRaisesRegex(RuntimeError, 'cannot retry'):
            pipe.prepare_next()
        with self.assertRaisesRegex(RuntimeError, 'cannot retry'):
            pipe.acquire_next()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(pipe.summary()['async_submit_calls'], 1)
        self.assertEqual(pipe.summary()['submitted_reads'], 0)
        self.assertTrue(pipe.summary()['failed'])
        self.assertFalse(pipe.summary()['drained'])

    def test_collect_errors_are_terminal_and_ticket_is_retained(self):
        for result in (None, ([], 0), ([0], -1), ([0], 11), StopIteration('fixture')):
            attempts, ticket = [], object()
            def collect(value):
                attempts.append(value)
                if isinstance(result, Exception):
                    raise result
                return result
            pipe = WindowAccumulator([(range(10), 0)], lambda batches: ticket,
                                     collect, lambda batch: None, 0)
            self.assertTrue(pipe.prepare_next())
            with self.assertRaises((RuntimeError, TypeError)):
                pipe.acquire_next()
            self.assertIs(pipe.outstanding_ticket, ticket)
            self.assertEqual(pipe.delivered, 0)
            self.assertEqual(pipe.previous_cpu_per_batch, 0)
            with self.assertRaisesRegex(RuntimeError, 'cannot retry'):
                pipe.acquire_next()
            with self.assertRaisesRegex(RuntimeError, 'cannot retry'):
                pipe.prepare_next()
            self.assertEqual(attempts, [ticket])
            self.assertFalse(pipe.summary()['drained'])

    def test_sampling_and_hint_errors_do_not_submit_or_retry(self):
        def broken_source():
            yield (range(10), 0)
            raise OSError('sampling fixture')
        for mode in ('sampling', 'hint', 'hint_stop'):
            calls = []
            def hint(batch):
                calls.append('hint')
                if mode == 'hint':
                    raise ValueError('hint fixture')
                if mode == 'hint_stop':
                    raise StopIteration('hint fixture')
            source = broken_source() if mode == 'sampling' else [(range(10), 0)]
            pipe = WindowAccumulator(source, lambda batches: calls.append('submit'),
                                     lambda ticket: None, hint, 0)
            with self.assertRaises((OSError, ValueError, RuntimeError)):
                pipe.prepare_next()
            with self.assertRaisesRegex(RuntimeError, 'cannot retry'):
                pipe.prepare_next()
            self.assertEqual(calls, ['hint'])
            self.assertEqual(pipe.sampled, 1)
            self.assertFalse(pipe.summary()['drained'])


if __name__ == '__main__':
    unittest.main()
