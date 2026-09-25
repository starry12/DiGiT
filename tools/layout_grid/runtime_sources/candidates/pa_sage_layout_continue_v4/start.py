"""Preview or start fourteen points using an explicit corrected pilot review."""
import argparse
import json
import os
import subprocess
from .common import ROOT, OUT, PY, UNIT, NATIVE_SHA, verify, require


def command():
    return ['/usr/bin/systemd-run', '--unit=' + UNIT,
        '--property=Type=exec', '--property=WorkingDirectory=' + str(ROOT), '--property=UMask=0022',
        '--property=Restart=no', '--property=KillMode=control-group', '--property=TimeoutStopSec=90',
        '--property=LimitMEMLOCK=infinity', '--setenv=CUDA_VISIBLE_DEVICES=2',
        '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=LD_LIBRARY_PATH=' + str(ROOT / 'bam/build/lib'),
        PY, '-B', '-u', '-m', 'candidates.pa_sage_layout_continue_v4.queue']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    execution = verify()
    from .queue import cells
    points = cells()
    print(json.dumps(dict(command=command(), execute=args.execute, candidate_sha256=execution,
        native_candidate_sha256=NATIVE_SHA, remaining_points=[k for k, _ in points],
        calibration='Reuse all four accepted v3 runs after documented counter-semantics correction',
        new_calibration_workers=False, raw_ssd_writes=False, new_pool_readback=False,
        epochs=1, evaluation='disabled', independent_smoke_for_remaining_points=False), indent=2), flush=True)
    if args.execute:
        require(os.geteuid() == 0, 'Native device access needs administrator launch')
        require(not (OUT / 'status.json').exists(), 'Preserve existing continuation evidence')
        from .queue import prerequisites
        prerequisites()
        subprocess.run(command(), check=True)


if __name__ == '__main__':
    main()
