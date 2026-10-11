"""Preview, or queue a persistent isolated experiment with local sudo."""
import argparse
import datetime
import subprocess
from .common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    a = parser.parse_args()
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + str(os.getpid())
    unit = 'digit-pa-revpr-node-' + stamp
    output = OUT / 'runs' / stamp
    cmd = ['/usr/bin/systemd-run','--unit=' + unit,'--property=Type=exec',
           '--property=WorkingDirectory=' + str(ROOT),'--property=UMask=0022',
           '--property=Restart=no','--property=KillMode=control-group',
           '--property=TimeoutStopSec=90','--property=LimitMEMLOCK=infinity',
           '--setenv=CUDA_VISIBLE_DEVICES=' + GPU,'--setenv=PYTHONDONTWRITEBYTECODE=1',
           '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=16',
           '--setenv=MKL_NUM_THREADS=16','--setenv=LD_LIBRARY_PATH=' + str(ROOT / 'bam/build/lib'),
           PY,'-B','-u','-m',MODULE + '.controller','--output',str(output)]
    plan = dict(command=cmd,unit=unit + '.service',output=str(output),execute=a.execute,
        arms=cfg()['ablation_order'],epoch_per_full_arm=1,full_updates_per_arm=1179,
        cpu_rows=cfg()['cpu_cache_rows'],cpu_feature_bytes=cfg()['cpu_cache_rows']*512,
        gpu_feature_bytes=cfg()['gpu_cache_bytes'],page_bytes=4096,
        graph='unchanged AE bidirectional',wait_for_active_reviewers=True,
        evaluation=False,raw_ssd_writes=False,
        source_sha256=sha(HERE/'manifest.json'))
    print(json.dumps(plan,indent=2),flush=True)
    if a.execute:
        require(os.geteuid()==0,'Run with local sudo')
        require(read(OUT/'cpu_checks.json')['passed'] and
                read(OUT/'cpu_checks.json')['source_sha256']==plan['source_sha256'],'Missing/stale CPU checks')
        subprocess.run(cmd,check=True)
        write(OUT/('launch_'+stamp+'.json'),plan)


if __name__=='__main__':
    main()
