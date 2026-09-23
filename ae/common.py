"""Current source/data validation, independent of experiment history."""
import hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import digit_paths
GIB = 2**30

def require(ok, message):
    if not ok: raise RuntimeError(message)

def read(path): return digit_paths.json_loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''): h.update(chunk)
    return h.hexdigest()
def binding(path): return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)
def append_sync(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def host(): return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024

def verify_release():
    from artifact_integrity import verify_component
    return verify_component(ROOT/'ae')

def device_idle():
    proc=subprocess.run(['fuser','/dev/libnvm0'],capture_output=True,text=True,timeout=15)
    require(proc.returncode==1 and not proc.stdout.strip(),'SSD busy or device query failed')

def check_payload(name):
    cfg=read(ROOT/'configs/ssd_payloads.json')[name]
    require(cfg['full_readback_passed'],'No complete payload readback')
    require(sha(cfg['state'])==cfg['state_sha256'],'SSD state changed; reverify payload')
    state=read(cfg['state'])
    require(state['status']=='verified' and state['device']=='/dev/libnvm0','Unverified/wrong SSD state')
    for key,expected in [('device_offset_bytes',cfg['offset']),('payload_bytes',cfg['bytes']),('feature_file_sha256',cfg['file_sha256'])]:
        require(state[key]==expected,'SSD state mismatch: '+key)
    return cfg

def check_device():
    # Identify is read-only. Never write or repopulate raw SSD automatically.
    require(os.geteuid()==0,'BaM device mapping requires root')
    require(Path('/dev/libnvm0').exists(),'Missing libnvm device')
    device_idle()
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'third_party/bam/build/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    p=subprocess.run([str(ROOT/'ae/native/identity/identify-module'),'--ctrl','/dev/libnvm0','--ns','1'],capture_output=True,text=True,timeout=60,check=True,env=env)
    import re
    cfg=read(ROOT/'configs/device.json')
    def field(name):
        match=re.search(r'^'+re.escape(name)+r'\s*:\s*(.+)$',p.stdout,re.M)
        require(match is not None,'Missing device identity field: '+name); return match.group(1).strip()
    require(field('Serial Number')==cfg['serial'],'Wrong SSD serial')
    require(int(field('Namespace identifier'),16)==cfg['namespace'],'Wrong namespace')
    block=int(field('Logical block size').split()[0])
    require(block==cfg['logical_block_bytes'] and int(field('Namespace capacity').split()[0])*block==cfg['capacity_bytes'],'Wrong SSD geometry')
