from .common import *

def main():
    from candidates.uks_group_first_v1.gpu_check import main as check
    from candidates.uks_group_first_v1.common import OUT as parent_out
    source=verify();check()
    r=read(parent_out/'gpu_checks.json')
    require(r['passed'] and r['source_sha256']==verify_first(),'Group-first GPU check failed')
    write(OUT/'gpu_checks.json',dict(passed=True,source_sha256=source,
          parent_receipt_sha256=sha(parent_out/'gpu_checks.json'),
          selection_policy=POLICIES['first'],native_training_accepted=False,raw_ssd_access=False))

if __name__=='__main__':main()
