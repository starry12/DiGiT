"""Bind accepted cache/graph provenance and validate full-epoch root scheduling."""
import ast
import json
import unittest
from .common import ROOT,HERE,OUT,SOURCE_RUN,require,sha,write,verify,load_roots,digest_array

def main():
    require(not (HERE/'manifest.json').exists(),'Already frozen')
    from candidates.cache_epoch_v1.common import verify as parent_verify
    from candidates.uks_freq_bfs_v1.common import hot_path,DATA,LARGE,OUT as BFS_OUT
    parent=parent_verify();OUT.mkdir(exist_ok=True)
    state=json.loads((SOURCE_RUN/'status.json').read_text())
    require(state['passed'] and state['complete'] and state['source_sha256']==parent,'Parent run not accepted')
    reports={arm:json.loads((SOURCE_RUN/('uks_'+arm)/'report.json').read_text()) for arm in ('gids','digit')}
    hot_sets={}
    for arm,path in [('gids',hot_path('gids')),('digit',SOURCE_RUN/'uks_digit/full_epoch_hot.npy')]:
        require(sha(path)==reports[arm]['hot_file_sha256'],'Hot set changed')
        hot_sets[arm]=dict(path=str(path),sha256=sha(path))
    canonical=LARGE/'canonical.npy';train=DATA/'synthetic/train.npy'
    receipt=json.loads((BFS_OUT/'bfs_receipt.json').read_text())
    require(sha(canonical)==receipt['canonical_sha256'],'BFS input changed')
    profile=reports['digit']['cache_epoch']['profile']
    binding=json.loads((ROOT/'results/uks_native_20260929_v1/binding.json').read_text())
    require(sha(train)==binding['files'][str(train)]['sha256'],'Training split changed')
    write(HERE/'inputs.json',dict(hot_sets=hot_sets,canonical=dict(path=str(canonical),sha256=sha(canonical)),
        training=dict(path=str(train),sha256=sha(train)),nodes=133633040,train_count=13363304,
        roots_sha256=profile['roots_sha256'],initial_model_sha256=reports['digit']['initial_model_sha256']))
    from .tests import Checks
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    require(result.wasSuccessful(),'Tests failed')
    # Real full roots must preserve the accepted 320-batch prefix for both arms.
    from candidates.uks_group_compare_v2.common import matched_roots
    import numpy as np
    roots=load_roots('gids');require(np.array_equal(roots[:320*1024],matched_roots('gids')),'Bounded prefix changed')
    require(digest_array(roots)==profile['roots_sha256'],'Profile/formal full-root mismatch')
    files={}
    for path in HERE.iterdir():
        if path.suffix not in ('.py','.json','.md'):continue
        if path.suffix=='.py':ast.parse(path.read_text(),str(path))
        files[str(path.relative_to(ROOT))]=sha(path)
    for path in (SOURCE_RUN/'status.json',SOURCE_RUN/'uks_digit/report.json',SOURCE_RUN/'uks_gids/report.json'):
        files[str(path.relative_to(ROOT))]=sha(path)
    write(HERE/'manifest.json',dict(parent_source=parent,files=files))
    source=verify();write(OUT/'cpu_checks.json',dict(passed=True,tests=result.testsRun,source_sha256=source,
        real_full_roots_checked=True,roots=len(roots),batches=13051,tail=104,native_accepted=False))
    print('Frozen '+source)

if __name__=='__main__':main()
