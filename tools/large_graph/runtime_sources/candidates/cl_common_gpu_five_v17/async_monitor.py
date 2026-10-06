"""One bounded background task per slow source; host guard never awaits it."""
import threading
import time
from collections import deque


class MonitorFailure(RuntimeError):
    pass


class PeriodicTask:
    def __init__(self, call, interval, *, clock=time.monotonic, max_events=128):
        self.call, self.interval, self.clock = call, interval, clock
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.started = self.active_since = None
        self.latest = None
        self.first_error = None
        self.events = deque(maxlen=max_events)
        self.dropped_events = 0

    def start(self):
        with self.lock:
            if self.thread is not None:
                raise RuntimeError('Background monitor already started')
            self.started = self.clock()
            self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.is_set():
            begin = self.clock()
            with self.lock:
                self.active_since = begin
            value, error, timings = None, None, None
            try:
                value = self.call()
            except BaseException as exc:
                error = repr(exc)
                timings = getattr(exc, 'query_timings', None)
            end = self.clock()
            row = dict(time=time.time(), started_monotonic=begin,
                       completed_monotonic=end, seconds=end-begin,
                       sample=value, error=error)
            if timings is not None:
                row['query_timings'] = timings
            with self.lock:
                self.active_since = None
                self.latest = row
                if error is not None and self.first_error is None:
                    self.first_error = row
                if len(self.events) == self.events.maxlen:
                    self.dropped_events += 1
                self.events.append(row)
            if error is not None:
                break
            if self.stop_event.wait(self.interval):
                break

    def inspect(self):
        now = self.clock()
        with self.lock:
            latest = self.latest
            return dict(started=self.started, latest=latest,
                        first_error=self.first_error,
                        initial_wait_seconds=None if self.started is None else now-self.started,
                        age_seconds=None if latest is None else now-latest['completed_monotonic'],
                        active_seconds=None if self.active_since is None else now-self.active_since,
                        dropped_events=self.dropped_events,
                        thread_alive=self.thread is not None and self.thread.is_alive())

    def drain(self):
        with self.lock:
            rows = list(self.events)
            self.events.clear()
        return rows

    def stop(self):
        # The read-only task may be inside a bounded subprocess; never join it
        # from the safety loop. No more work is scheduled after this call.
        self.stop_event.set()


class GPUChecks:
    def __init__(self, task, freshness=5.5):
        self.task, self.freshness = task, freshness

    def check(self):
        state = self.task.inspect()
        if state['started'] is None:
            raise MonitorFailure('GPU monitor not started')
        if state['dropped_events']:
            raise MonitorFailure('GPU monitor evidence queue overflow')
        if state['first_error'] is not None:
            raise MonitorFailure('GPU query failed: '+state['first_error']['error'])
        if state['active_seconds'] is not None and state['active_seconds'] > self.freshness:
            raise MonitorFailure('GPU query deadline exceeded')
        if state['latest'] is None:
            if state['initial_wait_seconds'] > self.freshness:
                raise MonitorFailure('GPU initial observation stale')
            return None
        if state['latest']['seconds'] > self.freshness:
            raise MonitorFailure('Completed GPU query exceeded deadline')
        if state['age_seconds'] > self.freshness:
            raise MonitorFailure('GPU observation stale')
        if not state['thread_alive'] and not self.task.stop_event.is_set():
            raise MonitorFailure('GPU monitor unexpectedly exited')
        return state['latest']['sample']


def check_runtime_decision(decision, gpu_checks):
    """The actual controller uses this ordering: no slow work after host alarm."""
    if not decision['ok']:
        raise RuntimeError('Runtime guard: '+str(decision['reasons']))
    return gpu_checks.check()


class AttributionTask:
    """Diagnostic provenance; failures/staleness are visible, never safety truth."""
    def __init__(self, path, unit, observer_factory, interval=5):
        self.path, self.unit, self.factory = path, unit, observer_factory
        self.lock = threading.Lock()
        self.memory = {}
        self.task = PeriodicTask(self._sample, interval)

    def update(self, memory):
        with self.lock:
            self.memory = dict(memory)

    def _sample(self):
        with self.lock:
            memory = dict(self.memory)
        # This thread alone owns the diagnostic streams. A delayed task cannot
        # write through a handle closed by the main controller.
        with self.path.open('a') as stream:
            if not hasattr(self, 'observer'):
                self.observer = self.factory(stream)
                self.observer.unit = self.unit
            self.observer.log = stream
            value = self.observer.sample(memory)
        return dict(collection_seconds=value.get('collection_seconds'))

    def start(self):
        self.task.start()
        return self

    def inspect(self):
        value = self.task.inspect()
        age = value['age_seconds']
        value['diagnostic_only'] = True
        elapsed=age if age is not None else value['initial_wait_seconds']
        value['stale'] = elapsed is None or elapsed > 15
        value['failed'] = value['first_error'] is not None
        return value

    def stop(self):
        self.task.stop()
