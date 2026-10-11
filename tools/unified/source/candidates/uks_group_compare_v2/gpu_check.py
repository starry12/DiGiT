"""Exercise the v2-generated overlay before any native SSD smoke."""
from .common import *

def main():
    from .overlay_gpu_check import run
    source=verify()
    run(OUT/'overlay_gpu_checks.json')
    result=read(OUT/'overlay_gpu_checks.json')
    require(result['passed'] and result['source_sha256']==v2.verify(),'v2 GPU check failed')
    write(OUT/'gpu_checks.json',dict(passed=True,source_sha256=source,
          parent_receipt_sha256=sha(OUT/'overlay_gpu_checks.json'),
          native_training_accepted=False,raw_ssd_access=False))

if __name__=='__main__':main()
