import subprocess
from candidates.uks_group_compare_v2.resources import target_gpu_busy
from .common import ROOT,read,require

def prior_controller_busy():
    # Wait for the whole prior chain, including idle gaps between its workers.
    p=ROOT/'results/uks_group_compare_20261010_v2/launch.json'
    if not p.exists():return []
    unit=read(p)['unit']
    require(unit.startswith('digit-uks-group-compare-v2-') and unit.endswith('.service') and '/' not in unit,'Unexpected prior service')
    state=subprocess.check_output(['systemctl','show',unit,'-p','ActiveState','--value'],text=True,timeout=30).strip()
    require(state in ('active','activating','deactivating','inactive','failed'),'Unknown prior service state')
    return [unit+': '+state] if state in ('active','activating','deactivating') else []
