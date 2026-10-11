"""Tiny CUDA cache-lookup and grouped sampler/model checks; no raw SSD access."""
import tempfile
import subprocess
from .common import *


def main():
    require(not (OUT/'gpu_checks.json').exists(),'Preserve GPU evidence')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Use physical GPU 2')
    query=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used',
                                   '--format=csv,noheader,nounits'],text=True).strip().split(',')
    require(query[0].strip()==GPU_UUID and int(query[1])<1024,'GPU 2 is occupied or differs')
    setup()
    import numpy as np
    import torch
    import dgl
    import BAM_Feature_Store as bam
    from candidates.pa_sage_ablation_cache_v3.cache import logical_slots,map_storage
    from .tests import fixture,stub_loader
    from candidates.pa_sage_512b_pair_v1.runtime_helpers import verify_digit_blocks
    from digit.sampler import DiGiTNeighborSampler,DIGIT_STORAGE_ROW
    from models import SAGE
    require(Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()==BINARY,'Wrong backend')
    began=time.time();cases=[]
    require(getattr(bam,'FIFO_WINDOW_512B_API',0)==1,'Missing FIFO/window ABI')
    fifo_cases=[]
    for repeat in range(3):
        for mode in range(4):
            result=list(bam.fifo_window_probe(mode))
            require(result[0]==(0 if mode==3 else 1),'Wrong victim or protected page evicted')
            require(result[1:3]==([1,129] if mode==0 else [0,0]),'Window protection was not acquired')
            require(result[3:]==[0,0,0,4,4,4,3],'Reference drain, eviction or ticket mismatch')
            fifo_cases.append(dict(repeat=repeat,mode=mode,values=result))
    for seed in range(8):
        rng=np.random.RandomState(seed);n=31;hot=np.sort(rng.choice(n,11,replace=False))
        storage=rng.randint(-1,n,size=257)
        storage[:10]=[hot[0],-1,next(x for x in range(n) if x not in hot),hot[1],hot[0],-1,hot[2],hot[0],hot[0],hot[1]]
        mapping=map_storage(storage,logical_slots(hot,n,np),np)
        features=np.arange(n*128,dtype=np.float32).reshape(n,128)
        requests=np.arange(len(storage)+2,dtype=np.uint64)
        actual=np.asarray(bam.exact_cpu_probe(mapping.tolist(),features[hot].reshape(-1).tolist(),requests.tolist(),128)).reshape(-1,128)
        expected=np.full_like(actual,-1)
        for i,row in enumerate(requests):
            if row<len(storage) and storage[row] in hot:expected[i]=features[storage[row]]
        np.testing.assert_array_equal(actual,expected);cases.append(seed)
    invalid=np.asarray(bam.exact_cpu_probe([0,1,3,4294967295],list(range(256)),[0,1,2,3,4],128)).reshape(-1,128)
    np.testing.assert_array_equal(invalid[[0,2,3,4]],-np.ones((4,128)))
    np.testing.assert_array_equal(invalid[1],np.arange(128))
    with tempfile.TemporaryDirectory(prefix='digit-512b-gpu-') as folder:
        graph,arrays,bundle,features=fixture(folder,pin=True)
        torch.manual_seed(0);dgl.seed(0)
        sampler=DiGiTNeighborSampler([4,3,2],bundle,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64',random_seed=0)
        sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
        loader=stub_loader(bundle.arrays['reordered_features'],'explicit','cuda:0')
        hinted=[];read_rows=[]
        loader.window_buffering=lambda b:hinted.append(loader.resolve_batch_feature_rows(b).clone())
        def fake_read(bs):
            outputs=[]
            for b in bs:
                rows=loader.resolve_batch_feature_rows(b);read_rows.append(rows.clone())
                outputs.append(torch.tensor(np.asarray(bundle.arrays['reordered_features'][rows.cpu().numpy()]),device='cuda'))
            return outputs,0
        loader._read_many=fake_read
        source=iter(sampler.sample_blocks(graph,torch.arange(9,device='cuda')[lo:lo+4]) for lo in range(0,9,4))
        model=SAGE(128,128,2,3,.2).cuda();opt=torch.optim.Adam(model.parameters(),lr=.01)
        seen=[]
        while True:
            try:inp,out,blocks,x=loader.fetch_feature(128,source,'cuda:0')
            except StopIteration:break
            verify_digit_blocks(blocks,arrays,[4,3,2])
            rows=blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy()
            np.testing.assert_array_equal(bundle.arrays['storage_to_node'][rows],inp.cpu().numpy())
            np.testing.assert_array_equal(x.cpu().numpy(),features[inp.cpu().numpy()])
            logits=model([b.int().to('cuda') for b in blocks],x)
            loss=torch.nn.functional.cross_entropy(logits,out%2);opt.zero_grad();loss.backward();opt.step()
            require(torch.isfinite(loss).item(),'Nonfinite GPU loss');seen.extend(out.cpu().tolist())
        require(seen==list(range(9)) and sampler._cuda_call_counter==3 and loader.pipeline.summary()['drained'],'GPU tail missing')
        require(len(hinted)==len(read_rows)==3,'Lost window hint/read')
        for h,r in zip(hinted,read_rows):require(torch.equal(h,r),'Hint/read address differs')
        graph._graph.unpin_memory_()
    files=[HERE/'gpu_tests.py',HERE/'tests.py',HERE/'common.py',HERE/'runtime/GIDS/GIDS.py']
    write(OUT/'gpu_checks.json',dict(passed=True,cache_random_seeds=cases,invalid_slot_rejected=True,
        fifo_window_cases=fifo_cases,fifo_window_api=1,
        grouped_cuda_batches=3,model_updates=3,features_bit_exact=True,storage_inverse_exact=True,
        explicit_eids_pinned=True,window_tail_drained=True,binary_sha256=sha(BINARY),parent_sha256=parent.verify(),
        seconds=time.time()-began,raw_ssd_access=False,full_native_read_tested=False,
        tested_sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    print('Tiny GPU checks passed; native SSD smoke and full epochs remain pending',flush=True)


if __name__=='__main__':main()
