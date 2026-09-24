"""Mount fixed prepared data read-only; never generate data or write raw SSD."""
import json, os, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREFIX = Path('/srv/digit-ae/releases/pa_sage_ablation_20260924_v1')
ALLOWED = {'data/papers_g2_random_v2', 'data/pa_sage_bidir_native_v1', 'ssd_state'}
def main():
    assert os.geteuid() == 0
    mounts = json.loads((HERE / 'mounts.json').read_text())
    assert {str(Path(m['destination']).relative_to(PREFIX)) for m in mounts} == ALLOWED
    for m in mounts:
        src, dst = Path(m['source']), Path(m['destination'])
        assert src.is_absolute() and src.is_dir() and not dst.is_symlink()
        dst.mkdir(parents=True, exist_ok=True)
        active = subprocess.run(['/usr/bin/mountpoint', '-q', str(dst)]).returncode == 0
        if not active:
            subprocess.run(['/usr/bin/mount', '--bind', str(src), str(dst)], check=True)
            subprocess.run(['/usr/bin/mount', '-o', 'remount,bind,ro', str(dst)], check=True)
        assert os.path.samefile(src, dst), 'Unexpected mounted source'
        assert os.statvfs(dst).f_flag & os.ST_RDONLY, 'Data is not read-only'
    print('Three fixed read-only mounts verified')
if __name__ == '__main__': main()
