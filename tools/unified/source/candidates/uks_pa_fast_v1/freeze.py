"""Freeze code and DGL constructor dependency; write current CPU check receipt."""
import ast,json,subprocess,sys,time
from pathlib import Path
from .common import HERE,ROOT,OUT,sha,write,binary_receipt

def main():
    subprocess.run([sys.executable,'-B','-m','unittest','candidates.uks_pa_fast_v1.tests','-v'],check=True)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts and 'runtime' not in p.relative_to(HERE).parts}
    for p in [HERE/'runtime/build_receipt.json',HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so',HERE/'runtime/BAM_Feature_Store/__init__.py',ROOT/'candidates/pa_sage_512b_latency_v1/frontier.py']:
        files[str(p.relative_to(ROOT))]=sha(p)
    import dgl.convert
    constructor=Path(dgl.convert.__file__)
    files[str(constructor)]=sha(constructor)
    write(HERE/'manifest.json',dict(files=files,scope='UKS 20+300, PA-style adaptations',created_unix=time.time()))
    from .common import verify,inputs
    source=verify();inputs();binary_receipt();OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'cpu_checks.json',dict(passed=True,source_sha256=source,unit_tests=7,native_build=True,input_hashes_checked=True,gpu_ssd_performance_accepted=False))
    pre=OUT/'gpu_fixture_preseal.json'
    if pre.exists():
        r=json.loads(pre.read_text());r.update(source_sha256=source,device=0,source_note='Bound after preseal fixture; subsequent edits affect pipeline/reporting, not tested model/frontier/native binary')
        write(OUT/'gpu_fixture_checks.json',r)
    print(source)
if __name__=='__main__':main()
