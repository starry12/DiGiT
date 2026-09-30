"""Preview by default; --execute starts a persistent isolated native service."""
import argparse
import datetime
import subprocess
from .common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    execution = verify()
    check_inputs(read(OUT / 'inputs.json'))
    require(read(OUT / 'cpu_checks.json')['passed'], 'CPU checks not passed')
    require(read(OUT/'gpu_checks.json')['passed'], 'Tiny GPU checks not passed')
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + str(os.getpid())
    unit = 'digit-pa-sage-512b-profile-v1-' + stamp
    folder = OUT / 'native' / stamp
    command = ['/usr/bin/systemd-run', '--unit=' + unit,
        '--property=Type=exec', '--property=WorkingDirectory=' + str(ROOT),
        '--property=UMask=0022', '--property=Restart=no', '--property=KillMode=control-group',
        '--property=TimeoutStopSec=90', '--property=LimitMEMLOCK=infinity',
        '--setenv=CUDA_VISIBLE_DEVICES=2', '--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=16', '--setenv=MKL_NUM_THREADS=16',
        '--setenv=LD_LIBRARY_PATH=' + str(ROOT / 'bam/build/lib'), PY, '-B', '-u', '-m',
        MODULE + '.control', '--output', str(folder)]
    plan = dict(command=command, unit=unit + '.service', output=str(folder), execute=args.execute,
        candidate_sha256=execution, train_count=1207179, expected_updates=1179,
        page_bytes=512, window_buffer=True, accumulator=True, epochs=1,
        evaluation=False, continuous_gpu_monitoring=False, raw_ssd_writes=False,
        full_order=full_schedule(), profile_modes=['off','host'], full_workers=8, bound_logical_cpu=2, block_index_bits=32)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        require(os.geteuid() == 0, 'Run this launcher using local sudo')
        subprocess.run(command, check=True)
        write(OUT / 'launch.json', dict(plan, started_unix=time.time()))


if __name__ == '__main__':
    main()
