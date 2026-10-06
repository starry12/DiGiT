"""Occasional GPU observations; query availability is diagnostic, not a stop rule."""
from .async_monitor import PeriodicTask, MonitorFailure


class GPUConflict(RuntimeError):
    """A successful query proved a GPU identity or ownership conflict."""


class GPUObservation:
    def __init__(self, monitor, interval=10):
        self.monitor = monitor
        self.task = PeriodicTask(self._poll, interval)
        self.last = None
        self.conflict = None

    def _poll(self):
        try:
            return dict(available=True, observation=self.monitor.poll())
        except GPUConflict as error:
            self.conflict = str(error)
            return dict(available=True, conflict=self.conflict)
        except Exception as error:
            # A timeout never terminates the background task or the worker.
            return dict(available=False, warning=repr(error),
                        query_timings=getattr(error, 'query_timings', None))

    def start(self):
        self.task.start()
        return self

    def check(self):
        for row in self.task.drain():
            yield row

    def inspect(self):
        state = self.task.inspect()
        if self.conflict:
            raise MonitorFailure(self.conflict)
        if state['first_error'] is not None:
            raise MonitorFailure('GPU observer internal failure: '+state['first_error']['error'])
        latest = state['latest']
        if latest is not None:
            self.last = latest['sample']
        if self.last and self.last.get('conflict'):
            raise MonitorFailure(self.last['conflict'])
        return dict(latest=self.last, age_seconds=state['age_seconds'],
                    query_active_seconds=state['active_seconds'], diagnostic_only=True)

    def stop(self):
        self.task.stop()
