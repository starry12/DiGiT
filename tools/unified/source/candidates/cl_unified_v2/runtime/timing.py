"""Explicit report contract: no stage hooks or additional stage barriers."""
from . import protocol as P
TIMING=dict(stage_profiling=False,scope='sum_of_common_training_windows',warmup_batches=P.WARMUP,
            measured_batches=P.MEASURED,boundary_synchronization='common_window_boundaries')
def valid_timing(report):return report.get('timing_protocol')==TIMING and 'stage_profile' not in report
