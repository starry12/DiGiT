import unittest
import numpy as np
from .common import *
from .binding import protocol
from .storage import pages_for
from .controller import accept
from .runtime import loader_kwargs,arm_budget,setup_sampling_imports
from .graph import edge_ids64,offsets64
class Checks(unittest.TestCase):
    def test_extents(self):
        p=protocol();endpoints=[]
        for arm,rows in [('gids',p['nodes']),('digit',read(DATA/'g2/receipt.json')['num_storage_rows'])]:
            lo=p['ssd_offsets'][arm];hi=lo+rows*1024
            self.assertEqual(lo%4096,0);self.assertEqual(rows%4,0)
            self.assertLessEqual(hi,read(ROOT/'configs/device.json')['capacity_bytes'])
            for path in (ROOT/'ssd_state').glob('libnvm0*.json'):
                v=read(path);old=v.get('device_offset_bytes',0);size=v['payload_bytes']
                self.assertTrue(hi<=old or lo>=old+size,str(path))
            for x,y in endpoints:self.assertTrue(hi<=x or lo>=y)
            endpoints.append((lo,hi));self.assertEqual((rows//4)%pages_for(rows//4),0)
    def test_geometry(self):
        setup_sampling_imports()
        from digit.io_geometry import IOGeometry
        p=protocol();g=IOGeometry.create(feature_row_bytes=1024,group_size=2,minimum_transfer_bytes=4096,target_request_bytes=4096)
        for arm in ARMS:
            k=loader_kwargs(p,arm,100,dict(feature_mode='logical_synthetic',passed=True,row_bytes=1024,offset=4*2**40,verified_bytes=100*1024),g)
            self.assertEqual(k['cache_dim'],256);self.assertEqual(k['cache_size'],4096);self.assertEqual(k['cpu_feature_path'],'mapped');self.assertEqual(arm_budget(p,arm)['cpu_feature_bytes'],p['cpu_cache_rows']*1024)
    def test_eid64(self):
        ptr=np.array([0,2**32+5,2**32+9],dtype=np.int64);offsets64(ptr,2**32+9)
        eid=np.array([2**32+4,2**32+5],dtype=np.int64);edge_ids64(eid,2**32+9)
        np.testing.assert_array_equal(np.searchsorted(ptr,eid,side='right')-1,[0,1])
    def test_accept_requires_normal_exit_and_independent_profile(self):
        r=dict(passed=True,source_sha256='x',batches=100,seed=23,feature_reads=0,optimizer_updates=0)
        self.assertEqual(accept('profile',r,0,'x'),r)
        for rc in [-6,1]:
            with self.assertRaises(Exception):accept('profile',r,rc,'x')
        with self.assertRaises(Exception):accept('profile',dict(r,optimizer_updates=1),0,'x')
        with self.assertRaises(Exception):accept('profile',r,0,'changed')
    def test_small_graph_feature_mapping_and_backward(self):
        import tempfile
        from pathlib import Path
        setup_sampling_imports()
        import torch,dgl
        from digit.reorganization import reorganize_to_bundle
        from digit.sampler import DiGiTNeighborSampler,DIGIT_STORAGE_ROW
        from .graph import normalized_fixture,graph_from_csc
        from .model import create
        torch.set_num_threads(1)
        n=32;p=protocol();p.update(nodes=n,fixture=True)
        raw=np.array([(i,(i+j)%n) for i in range(n) for j in (0,1,3,7)],dtype=np.int64).T
        ptr,idx,eid=normalized_fixture(raw,n);graph=graph_from_csc(ptr,idx,eid,n,fixture=True)
        features=np.random.default_rng(4).normal(size=(n,256)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            artifact=reorganize_to_bundle(ptr,idx,features,Path(tmp)/'g2',dataset_name='UKS',dataset_size='fixture',group_size=2,replication_ratio=.2,page_size=4096,minimum_transfer_bytes=4096,target_request_bytes=4096,hot_nodes=np.array([0,1,2],dtype=np.int64),seed=0)
            samplers=[dgl.dataloading.NeighborSampler([3,2,2]),DiGiTNeighborSampler([3,2,2],artifact,cuda_mode='disabled',metadata_mode='cpu_eid',random_seed=0)]
            for j,sampler in enumerate(samplers):
                inp,out,blocks=sampler.sample_blocks(graph,torch.tensor([0,7,15,31]))
                x=features[inp.numpy()].copy()
                if j:
                    rows=blocks[0].srcdata[DIGIT_STORAGE_ROW].numpy()
                    np.testing.assert_array_equal(artifact.arrays['reordered_features'][rows],x)
                for block in blocks:
                    u,v=block.edges(order='eid');ids=block.edata[dgl.EID].numpy()
                    np.testing.assert_array_equal(block.srcdata[dgl.NID][u].numpy(),idx[ids])
                    np.testing.assert_array_equal(block.dstdata[dgl.NID][v].numpy(),np.searchsorted(ptr,ids,side='right')-1)
                net,opt,initial=create(p);loss=torch.nn.functional.cross_entropy(net(blocks,torch.from_numpy(x)),torch.tensor([0,1,2,3]))
                loss.backward();self.assertTrue(torch.isfinite(loss).item());opt.step()
                from candidates.pa_sage_cache_policy_v1.training import model_hash
                self.assertNotEqual(model_hash(net),initial)
        self.assertFalse(torch.cuda.is_initialized())
    def test_import_isolation(self):
        import sys
        self.assertNotIn('GIDS',sys.modules)
        self.assertFalse(any(x.startswith('candidates.') and 'DiGiTSamplerCUDA' in x for x in sys.modules))
if __name__=='__main__':unittest.main()
