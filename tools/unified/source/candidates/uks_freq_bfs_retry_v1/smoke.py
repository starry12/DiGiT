import numpy as np
from .common import *
from candidates.uks_native_v1.binding import protocol,check
from .common import select_sampler as graph_sampler
def smoke(arm,folder):
    from candidates.uks_mixed_1k2k_v1.runtime import prepare_imports,loader_kwargs,arm_budget
    cls,module=prepare_imports()
    import torch,dgl
    from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP
    from digit.io_geometry import IOGeometry
    from candidates.uks_mixed_1k2k_v1.backend import install_on_loader
    from candidates.uks_mixed_1k2k_v1.counters import snapshot,interval
    from candidates.uks_native_v1.model import create
    from candidates.ig_sage_affinity_pair_v1.affinity import apply_affinity
    apply_affinity('digit_full' if arm=='digit' else 'gids')
    p=protocol();b=check();graph,arrays,artifact,sampler=graph_sampler(arm,0)
    geometry=IOGeometry.create(feature_row_bytes=1024,group_size=1,minimum_transfer_bytes=1024,target_request_bytes=1024)
    rows=p['nodes'] if arm=='gids' else b['receipts']['g2']['num_storage_rows']
    from candidates.uks_native_v1.storage import api,state_path
    v=load_plan(arm);api()._validate_active_state(state_path(v),v,'verified')
    # load_plan validates the saved full readback receipt and current input identities.
    pool=dict(passed=True,feature_mode='logical_synthetic',row_bytes=1024,offset=v.device_offset_bytes,verified_bytes=v.payload_bytes)
    loader=cls(**loader_kwargs(p,arm,rows,pool,geometry))
    if arm=='gids':
        hot=np.load(hot_path(arm),mmap_mode='r');primary=np.arange(p['nodes'],dtype=np.int64);storage=primary
    else:
        hot=np.load(hot_path(arm),mmap_mode='r');primary=artifact.arrays['node_to_primary_row'];storage=artifact.arrays['storage_to_node']
    installed=install_on_loader(loader,module,arm_budget(p,arm),hot,primary,storage)
    net,opt,initial=create(p,device='cuda');features=np.load(DATA/'synthetic/features.npy',mmap_mode='r');labels=np.load(DATA/'synthetic/labels.npy',mmap_mode='r')
    roots=load_roots(arm)[:64*1024].copy();rootgpu=torch.from_numpy(roots).cuda()
    def batches():
        for i in range(64):
            inp,out,blocks=sampler.sample_blocks(graph,rootgpu[i*1024:(i+1)*1024])
            if arm=='gids':
                blocks[0].srcdata[DIGIT_STORAGE_ROW]=inp
                blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP]=torch.zeros_like(inp,dtype=torch.bool)
            yield inp,out,blocks
    it=iter(batches());loader.BAM_FS.begin_useful_io_region();before=snapshot(loader.BAM_FS);logical=0;losses=[]
    for i in range(64):
        inp,out,blocks,x=loader.fetch_feature(256,it,torch.device('cuda:0'))
        require(torch.equal(out,rootgpu[i*1024:(i+1)*1024]),'Changed output roots')
        ids=inp.cpu().numpy();wanted=np.array(features[ids],copy=True);got=x.cpu().numpy()
        require(np.array_equal(wanted.view(np.uint32),got.view(np.uint32)) and np.isfinite(got).all(),'SSD features not bit exact')
        for block in blocks:
            u,w=block.edges(order='eid');src=block.srcdata[dgl.NID][u].cpu().numpy();dst=block.dstdata[dgl.NID][w].cpu().numpy();eid=block.edata[dgl.EID].cpu().numpy()
            require(np.all((eid>=0)&(eid<len(arrays[1]))),'EID out of range')
            require(np.array_equal(src,arrays[1][eid]) and np.array_equal(dst,np.searchsorted(arrays[0],eid,side='right')-1),'Sample edge differs from normalized CSC')
        pred=net(blocks,x);target=torch.from_numpy(np.array(labels[roots[i*1024:(i+1)*1024]],copy=True)).cuda();loss=torch.nn.functional.cross_entropy(pred,target)
        opt.zero_grad(set_to_none=True);loss.backward()
        require(torch.isfinite(loss).item() and all(v.grad is None or torch.isfinite(v.grad).all().item() for v in net.parameters()),'Nonfinite training')
        opt.step();losses.append(loss.item());logical+=len(inp);write(folder/'progress.json',dict(stage='short_training',arm=arm,updates=i+1))
        region=interval(before,snapshot(loader.BAM_FS),logical,complete_region=True)
        write(folder/'coverage.json',dict(updates=i+1,covered=coverage(region),region=region))
        if i+1>=4 and coverage(region):break
    torch.cuda.synchronize();region=interval(before,snapshot(loader.BAM_FS),logical,complete_region=True)
    require(region['policy']==p['arms'][arm]['gpu_policy'],'Measured replacement differs')
    require(coverage(region),'Short did not cover all cache/I/O paths')
    if arm=='gids':require(region['request_sizes']['primary_2k']==0,'GIDS unexpectedly merged reads')
    from candidates.pa_sage_cache_policy_v1.training import model_hash
    final=model_hash(net);require(initial!=final,'Optimizer did not update')
    require(all(torch.isfinite(v).all().item() for v in net.parameters()),'Nonfinite final model')
    return dict(passed=True,arm=arm,updates=i+1,examples=(i+1)*1024,feature_bit_exact=True,sample_edges_verified=True,roots_sha256=digest(roots[:(i+1)*1024]),losses=losses,initial_model_sha256=initial,final_model_sha256=final,region=region,cache=installed,native=True)
