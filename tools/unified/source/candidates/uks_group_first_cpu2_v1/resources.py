"""Wait for the preceding PA/UKS controller, including gaps between workers."""
import subprocess
from candidates.uks_group_compare_v2.resources import target_gpu_busy


def prior_controller_busy():
    result=subprocess.run(['systemctl','list-units','digit-pa*','digit-uks*','--all',
        '--no-pager','--plain','--no-legend'],capture_output=True,text=True,timeout=30,check=True)
    active=[]
    for line in result.stdout.splitlines():
        fields=line.split()
        if len(fields)>=4 and fields[2] in ('active','activating','deactivating') and fields[3]!='exited':
            active.append(fields[0]+': '+fields[2])
    return active
