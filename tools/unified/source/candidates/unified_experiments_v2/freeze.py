"""Seal isolated v2 source/binaries while retaining inherited dependency pins."""
import ast,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SHARED=('unified_training_v2','unified_results_v2','large_graph_fast_v1')
BACKENDS=('ig_unified_v2','uks_unified_v2','ukl_unified_v2','cl_unified_v2')
def files_for(names):
    files=[]
    for name in names:
        for p in (ROOT/'candidates'/name).rglob('*'):
            if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.txt','.h','.hpp','.cuh','.cu','.cpp','.so'):
                if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
                files.append(p)
    return files
def seal(name,files,extra=None):
    m=dict(extra or {},files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))},created_unix=time.time())
    (ROOT/'candidates'/name/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
def main():
    from candidates.large_graph_fast_v1.build import verify
    for k in ('ukl','cl'):
        for role in ('io','sampler'):verify(k,role)
    build=[p for p in (ROOT/'results/large_graph_fast_20261011_v1/build').glob('*') if p.suffix in ('.so','.json') and 'before_' not in p.name]
    for name in BACKENDS:
        m=json.loads((ROOT/'candidates'/name.replace('_v2','_v1')/'manifest.json').read_text())
        inherited=[ROOT/f for f in m['files'] if not any('/'+n.replace('_v2','_v1')+'/' in '/'+f for n in (*SHARED,name))]
        seal(name,files_for((*SHARED,name))+inherited+build,{k:v for k,v in m.items() if k not in ('files','created_unix')})
    seal('unified_experiments_v2',files_for((*SHARED,*BACKENDS,'unified_experiments_v2'))+build+
        [ROOT/'candidates'/n/'manifest.json' for n in BACKENDS]+
        files_for(('pa_sage_bfs256_short_v1',))+[ROOT/'candidates/pa_sage_bfs256_short_v1/manifest.json'])
    print('Unified v2 manifests sealed; GPU acceptance is separate.')
if __name__=='__main__':main()
