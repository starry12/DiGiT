from .common import *

def main():
    require(not (HERE/'manifest.json').exists(),'Preserve frozen source')
    checks=read(OUT/'cpu_checks.json');require(checks['passed'],'Missing CPU checks')
    # Pin transitive manifests and verify their full contents before freezing.
    from candidates.uks_group_compare_v2.common import verify as compare_verify
    compare_verify()
    deps=['candidates/uks_group_compare_v2/manifest.json',
          'candidates/uks_group_incremental_v1/native/sampler.cu',
          'candidates/pa_sage_bidir_native_v2/runtime/digit/sampler.py',
          'candidates/pa_sage_bidir_native_v2/runtime/digit/outer_selection.py']
    files=list(HERE.glob('*.py'))+list((HERE/'native').glob('*'))+[HERE/'README.md']
    write(HERE/'manifest.json',dict(files={str(p.relative_to(HERE)):sha(p) for p in files},
          dependencies={p:sha(ROOT/p) for p in deps}))
    print('Frozen '+verify())

if __name__=='__main__':main()
