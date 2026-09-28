"""Check metadata-only dependencies, including stat-only source-order provenance."""
import json
from pathlib import Path
ROOT=Path('/home/embed/digit');H=Path(__file__).resolve().parent;BASE=H.parent
R=Path('/srv/digit-ae/releases/ig_sage_performance_20260928_v1')
def read(p):return json.loads(Path(p).read_text())
def identity(p):
    s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def dependencies():
    paths=set(read(ROOT/'results/ig_perf_preflight_20260922_v5/check/inputs.json')['files'])
    for candidate in ('ig_perf_v5','ig_perf_window300_v1'):
        cfg=read(ROOT/'candidates'/candidate/'protocol.json')
        for key in ('train_order','source_train_order','source_order_ready'):paths.add(cfg[key])
        ready=read((ROOT/cfg['train_order']).parent/'ready.json')
        source=ROOT/cfg['source_train_order'];assert identity(source)==ready['source_identity'],'Source order changed'
        paths.add(str((ROOT/cfg['train_order']).parent/'ready.json'))
    return [p if p.is_absolute() else ROOT/p for p in map(Path,sorted(paths))]
def missing(mounts):
    snapshot=read(BASE/'control/snapshot_manifest.json')['files'];out=[]
    for path in dependencies():
        assert path.is_file(),str(path)
        if not ROOT in path.parents:
            assert str(path).startswith('/mnt/');continue
        n=str(path.relative_to(ROOT))
        covered=any(path==Path(m['source']) or Path(m['source']) in path.parents for m in mounts)
        if not covered and n not in snapshot:out.append(n)
    return out
if __name__=='__main__':
    old=read(BASE/'control/mounts.json');new=read(H/'mounts.json')
    before=missing(old);after=missing(new)
    assert 'data/ig_perf_v2/train_order.npy' in before,before
    assert not after,after
    value=dict(passed=True,dependency_paths=len(dependencies()),old_missing=before,new_missing=after,large_file_contents_rehashed=False,training_started=False)
    (H/'dependency_check.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
