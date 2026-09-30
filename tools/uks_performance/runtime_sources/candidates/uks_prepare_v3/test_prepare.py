import unittest,tempfile
from pathlib import Path
import numpy as np
from .prepare import build_csc,build_rank,build_g2,synthetic,payload
from .protocol import fixture_plan,compile_plan
from .graph import normalized_fixture
from .backend import allocation,ExactInstaller
from .mapping import install
from candidates.pa_sage_cache_policy_v2.tests import FakeStore
class Tests(unittest.TestCase):
    def test_native_preparation_matches_independent_graph_oracle(self):
        n=60;rng=np.random.default_rng(7);edges=rng.integers(0,n,(2,1200),dtype=np.int64)
        edges=np.concatenate((edges,edges[:,:10],np.array([[0,0,1],[0,0,1]],np.int64)),axis=1)
        for transposed in (False,True):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);source=root/'source.npy';np.save(source,np.ascontiguousarray(edges.T if transposed else edges))
                for name in ('csc','rank','g2','synthetic','payload'):(root/name).mkdir()
                c=build_csc(source,n,root/'csc',7);ptr,idx,ids=normalized_fixture(edges,n)
                np.testing.assert_array_equal(np.load(root/'csc/indptr.npy'),ptr);np.testing.assert_array_equal(np.load(root/'csc/indices.npy'),idx);np.testing.assert_array_equal(np.load(root/'csc/eids.npy'),ids)
                build_rank(root/'csc',n,root/'rank')
                # Independent dense reverse-PageRank reference including multiedges.
                adj=np.zeros((n,n));
                for d in range(n):np.add.at(adj[d],idx[ptr[d]:ptr[d+1]],1)
                adj/=adj.sum(axis=1)[:,None];score=np.full(n,1/n)
                for _ in range(20):score=.15/n+.85*adj.T.dot(score)
                np.testing.assert_allclose(np.load(root/'rank/revpr.npy'),score,rtol=1e-12)
                g=build_g2(root/'csc',root/'rank',n,root/'g2');self.assertTrue(g['expanded_adjacency_multiset_exact'])
                m=np.load(root/'g2/group_members.npy');owners=np.load(root/'g2/group_owner.npy');rp=np.load(root/'g2/reordered_indptr.npy');ri=np.load(root/'g2/reordered_indices.npy')
                for d in range(n):
                    expanded=[]
                    for x in ri[rp[d]:rp[d+1]]:expanded.extend([x] if x<n else m[x-n].tolist())
                    self.assertEqual(sorted(expanded),sorted(idx[ptr[d]:ptr[d+1]]))
                hot=np.load(root/'rank/hot_nodes.npy');self.assertFalse(np.isin(m,hot).any())
                p=fixture_plan(n,8);synthetic(p,root/'synthetic');q=payload(root/'synthetic',root/'g2',root/'payload');self.assertTrue(q['logical_aliases_bit_exact'])
    def test_legacy_fifo_exact_rows_and_capacity(self):
        p=compile_plan();self.assertEqual(p['arms']['gids']['gpu_policy'],'legacy');self.assertEqual(p['arms']['digit']['gpu_policy'],'fifo')
        hot=np.array([0,2],np.int64);primary=np.array([0,1,3,5],np.int64);storage=np.array([0,1,-1,2,0,3,2,-1],np.int64)
        for name in p['arms']:
            arm=dict(p['arms'][name],cpu_rows=2,gpu_feature_cache_bytes=4*2**30)
            self.assertEqual(allocation(arm)['gpu_dma_allocation_bytes'],4*2**30)
            fs=FakeStore();install(ExactInstaller(fs),arm,hot,primary,storage)
            self.assertEqual(fs.calls[-1],('configure',3 if name=='gids' else 2,4*2**30));self.assertEqual(fs.slots,[1,0,0,2,1,0,2,0])
    def test_bad_endpoint_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);np.save(root/'bad.npy',np.array([[0,70,1],[1,2,3]],np.int64));(root/'out').mkdir()
            import subprocess
            with self.assertRaises(subprocess.CalledProcessError):build_csc(root/'bad.npy',60,root/'out',7)
if __name__=='__main__':unittest.main(verbosity=2)
