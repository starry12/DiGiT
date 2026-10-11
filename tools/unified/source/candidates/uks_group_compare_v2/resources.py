"""Only GPU2 ownership blocks this user's comparison; other GPUs may be busy."""
import subprocess
from .common import require

GPU_UUID='GPU-927ce617-743a-4bfe-6a60-8a8311cfc703'

def parse_target_apps(text):
    rows=[]
    for line in text.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)==2 and parts[0]==GPU_UUID:rows.append(parts[1])
    return rows

def target_gpu_busy():
    uuid=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid',
                                  '--format=csv,noheader'],text=True,timeout=30).strip()
    require(uuid==GPU_UUID,'GPU2 UUID changed')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                  '--format=csv,noheader'],text=True,timeout=30)
    return parse_target_apps(apps)
