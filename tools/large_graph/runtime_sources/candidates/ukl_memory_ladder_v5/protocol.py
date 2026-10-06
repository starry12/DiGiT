import json,hashlib
from pathlib import Path
TIERS=(65536,98304)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/ukl_memory_ladder_20261002_v5'
PRIOR=ROOT/'results/ukl_memory_ladder_20261001_v4/completion_review.json'
REAL=ROOT/'results/ukl_real_fragment_20261001_v1/completion_review.json'
DATA=Path('/mnt/n0/digit/ukl_memory_ladder_v5')
def validate_tier(tier):
    if type(tier)!=int or tier not in TIERS:raise ValueError('Allowed tiers: 65536,98304 MiB only')
    return tier*1024**2

def budget(tier):
    size=validate_tier(tier);maximum=2*size+2*1024**3
    return dict(tier_mib=tier,arena_bytes=size,source_bytes=size,chunk_bytes=1024**2,gpu_buffer_bytes=1024**2,memory_max=maximum,memory_high=maximum-512*1024**2,memlock=size+128*1024**2,host_available_min=maximum+size+64*1024**3,disk_free_min=size+8*1024**3,runtime_seconds=(1800 if tier==65536 else 2400),source_rate_bytes_per_second=64*1024**2,source_fsync_bytes=16*1024**2,post_observation_seconds=180)

def prerequisite(tier):
    validate_tier(tier)
    manifest_sha=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()
    prior=json.loads(PRIOR.read_text())
    if not prior.get('passed') or [r['tier_mib'] for r in prior['tiers']]!=[16384,32768]:
        raise RuntimeError('Fourth phase incomplete')
    checks=[]
    for r in prior['tiers']:
        p=ROOT/r['receipt']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['receipt_sha256'] or not json.loads(p.read_text()).get('passed'):
            raise RuntimeError('Fourth-phase receipt changed')
        checks.append(dict(path=str(p),sha256=r['receipt_sha256']))
    for previous in TIERS[:TIERS.index(tier)]:
        runs=sorted(p for p in OUT.glob('tier_'+str(previous)+'_*') if p.is_dir())
        if not runs:raise RuntimeError('Earlier tier not run')
        p=runs[-1]/'acceptance.json'
        if not p.exists():raise RuntimeError('Earlier tier incomplete')
        r=json.loads(p.read_text())
        if not r.get('passed') or r.get('tier_mib')!=previous or r.get('manifest_sha256')!=manifest_sha:
            raise RuntimeError('Earlier tier failed or code changed')
        checks.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return checks
