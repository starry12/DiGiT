"""CPU-only worker: record one known unattributed GPU timeout, fail otherwise."""
import re
from candidates.ukl_native_sampling_v10r4.guard import KernelCursor as Base

TIMEOUT = re.compile(r'^NVRM: _threadNodeCheckTimeout: (?:_threadNodeCheckTimeout: )?(?:currentTime: [0-9a-fx]+ >= [0-9a-fx]+|Timeout was set to: 4000 msecs!)$', re.I)

class KernelCursor(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_classifications = []

    def _classify(self, records):
        candidates = super()._classify(records)
        self.cpu_classifications = []
        fatal = []
        for record in candidates:
            # Exact full-line match only. Unknown GPU failures, Xids, memory,
            # filesystem, NVMe and system-wide lockups retain the base policy.
            ignored = bool(TIMEOUT.fullmatch(record['MESSAGE']))
            self.cpu_classifications.append(dict(cursor=record['__CURSOR'],
                message=record['MESSAGE'], fatal=not ignored,
                reason='unattributed_gpu_thread_timeout_cpu_only' if ignored else 'base_kernel_alert'))
            if not ignored:
                fatal.append(record)
        return fatal

    def _result(self, query_ok, records, alerts, current_error=None):
        result = super()._result(query_ok, records, alerts, current_error)
        result.update(worker_scope='CPU_only_GPU_devices_denied',
            cpu_alert_classification=self.cpu_classifications if records else [])
        return result
