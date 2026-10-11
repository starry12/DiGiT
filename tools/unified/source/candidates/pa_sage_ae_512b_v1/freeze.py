"""CPU regression, runtime import and immutable dependency receipt."""
import unittest
from .common import *
def main():
    require(not (HERE/'manifest.json').exists(),'Preserve existing sealed candidate')
    from .tests import ComparisonTests
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonTests))
    require(result.wasSuccessful(),'CPU checks failed')
    setup()
    import torch,BAM_Feature_Store
    from digit import DiGiTSamplerCUDA
    from . import worker,training_loop,validation,controller,loader
    require(Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()==backend_binary(),'Wrong mixed read backend')
    require(Path(DiGiTSamplerCUDA.__file__).resolve()==PARENT/'runtime/digit/DiGiTSamplerCUDA.so','Changed AE sampler')
    require(BAM_Feature_Store.PA_PAIR_ASYNC_API==1,'Missing pair ABI')
    require(not torch.cuda.is_initialized(),'CPU readiness initialized CUDA')
    from candidates.pa_sage_bidir_native_v2.common import verify as parent
    from candidates.pa_sage_mixed_group_v1.common import verify as mixed
    files={}
    roots=(HERE,ROOT/'candidates/pa_sage_512b_pair_v3/runtime/GIDS')
    for tree in roots:
        for p in tree.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
                files[str(p.relative_to(ROOT))]=sha(p)
    for rel in ('ae/igb/bounded_io.py','candidates/pa_sage_revpr_node_v1/admission.py',
                'candidates/pa_sage_revpr_node_v1/common.py','candidates/pa_sage_mixed_bfs_v5/admission.py'):
        files[rel]=sha(ROOT/rel)
    write(HERE/'manifest.json',dict(files=files,ae_parent_sha256=parent(),mixed_backend_candidate_sha256=mixed(),
        deployed_protocol_sha256=sha(DEPLOYED/'candidates/pa_sage_bidir_native_v2/protocol.json')))
    source=verify()
    from .admission import estimate
    plans={arm:estimate(arm) for arm in cfg()['ablation_order']}
    write(OUT/'admission_plans.json',plans)
    write(OUT/'cpu_checks.json',dict(passed=True,source_sha256=source,tests=result.testsRun,native_import_verified=True,
        gpu_initialized=False,raw_ssd_access=False,full_training_accepted=False))
    print('Frozen '+source,flush=True)
    print('GPU admission GiB: '+str({k:round(v['required_bytes']/2**30,3) for k,v in plans.items()}))
if __name__=='__main__':main()
