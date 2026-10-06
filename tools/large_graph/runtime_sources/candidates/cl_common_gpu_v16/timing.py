"""Explicit report contract: no stage hooks or additional stage barriers."""
TIMING=dict(stage_profiling=False,scope='300_batch_end_to_end_wall',
            warmup_batches=20,measured_batches=300,
            boundary_synchronization='original_window_boundaries_only')

def valid_timing(report):
    return report.get('timing_protocol')==TIMING and 'stage_profile' not in report
