"""Default is plan only. Execute requires sudo and performs exact scoped writes."""
import argparse,json,os
from . import protocol as P

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    digest=P.verify_manifest();plan,bound,_=P.inputs(True)
    if not a.execute:
        print(json.dumps(dict(execute=False,manifest_sha256=digest,budget=P.budget(),arms=plan['arms'],
          sequence=['exclusive locks and idle check','device identity','fixed resume authorization and protected range checks','atomic reservations','guard fingerprints','full GIDS retained SHA256 and exact DiGiT prefix comparison','write only remaining DiGiT suffix with flushed checkpoints','flush and sample readback','guards and normal close','post-exit checks'],resume_offsets=P.authorization()['resume_offsets'],blank_scan_required=False,raw_ssd_written=False),indent=2));return
    if os.geteuid()!=0:raise RuntimeError('sudo required for bounded raw-device writer')
    from .monitor_only import main as qualify
    from .start import execute
    qualify();execute()
if __name__=='__main__':main()
