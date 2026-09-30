"""Fresh processes for two four-update native smoke checks; no SSD writes."""
import argparse,os,time,sys
from pathlib import Path
import numpy as np
from candidates.uks_native_v1.common import *
from candidates.uks_native_v1.binding import protocol,check,graph_sampler

def admission():
    from ae.common import host
    import subprocess
    q=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=30).strip().split(',')
    require(q[0].strip()=='GPU-927ce617-743a-4bfe-6a60-8a8311cfc703' and int(q[1])*2**20>=40*2**30,'GPU2 not free enough')
    require(host()>=256*2**30,'Need 256 GiB available for initial UKS native admission')

def smoke(arm,folder):
    from candidates.uks_native_v1.runtime import prepare_imports,loader_kwargs,arm_budget
    cls,module=prepare_imports()
    import torch,dgl
    from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP
    from digit.io_geometry import IOGeometry
    from candidates.uks_native_v1.backend import install_on_loader
    from candidates.uks_native_v1.counters import snapshot,interval
    from candidates.uks_native_v1.model import create
    p=protocol();b=check();graph,arrays,artifact,sampler=graph_sampler(arm,0)
    geometry=IOGeometry.create(feature_row_bytes=1024,group_size=2,minimum_transfer_bytes=4096,target_request_bytes=4096)
    rows=p['nodes'] if arm=='gids' else b['receipts']['g2']['num_storage_rows']
    from candidates.uks_native_v1.storage import plan,api,state_path
    v=plan(arm);api()._validate_active_state(state_path(v),v,'verified')
    from .receipts import validate_saved_receipt
    verified=validate_saved_receipt(OUT/('storage_'+arm)/api().VERIFY_RECEIPT,v)
    pool=dict(passed=True,feature_mode='logical_synthetic',row_bytes=1024,offset=v.device_offset_bytes,verified_bytes=v.payload_bytes)
    loader=cls(**loader_kwargs(p,arm,rows,pool,geometry))
    if arm=='gids':
        hot=np.load(DATA/'rank/hot_nodes.npy',mmap_mode='r');primary=np.arange(p['nodes'],dtype=np.int64);storage=primary
    else:
        profile_receipt=read(OUT/'profile/accepted.json');require(sha(LARGE/'freq_hot.npy')==profile_receipt['hot_sha256'],'Freq hot set changed')
        hot=np.load(LARGE/'freq_hot.npy',mmap_mode='r');primary=artifact.arrays['node_to_primary_row'];storage=artifact.arrays['storage_to_node']
    installed=install_on_loader(loader,module,arm_budget(p,arm),hot,primary,storage)
    net,opt,initial=create(p,device='cuda');features=np.load(DATA/'synthetic/features.npy',mmap_mode='r');labels=np.load(DATA/'synthetic/labels.npy',mmap_mode='r')
    roots=np.load(DATA/'synthetic/roots.npy',mmap_mode='r')[:4096].copy();rootgpu=torch.from_numpy(roots).cuda()
    def batches():
        for i in range(4):
            inp,out,blocks=sampler.sample_blocks(graph,rootgpu[i*1024:(i+1)*1024])
            if arm=='gids':
                blocks[0].srcdata[DIGIT_STORAGE_ROW]=inp
                blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP]=torch.zeros_like(inp,dtype=torch.bool)
            yield inp,out,blocks
    it=iter(batches());loader.BAM_FS.begin_useful_io_region();before=snapshot(loader.BAM_FS);logical=0;losses=[]
    for i in range(4):
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
    torch.cuda.synchronize();region=interval(before,snapshot(loader.BAM_FS),logical,complete_region=True)
    require(region['policy']==p['arms'][arm]['gpu_policy'],'Measured replacement differs')
    require(all(region['serving'][k]>0 for k in ('cpu_served_rows','gpu_hit_rows','ssd_served_rows')),'Short did not cover all cache/I/O paths')
    from candidates.pa_sage_cache_policy_v1.training import model_hash
    final=model_hash(net);require(initial!=final,'Optimizer did not update')
    require(all(torch.isfinite(v).all().item() for v in net.parameters()),'Nonfinite final model')
    return dict(passed=True,arm=arm,updates=4,examples=4096,feature_bit_exact=True,sample_edges_verified=True,roots_sha256=digest(roots),losses=losses,initial_model_sha256=initial,final_model_sha256=final,region=region,cache=installed,native=True)

def main():
    from .common import check_reuse,verify as repair_verify
    from ae.common import check_device
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['smoke_gids','smoke_digit']);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root GPU2 worker required')
    heavy_gate();check_reuse();admission();check_device();args.output.mkdir(parents=True,exist_ok=True)
    result=smoke(args.stage.split('_')[1],args.output)
    check_reuse();write(args.output/'report.json',dict(result,source_sha256=repair_verify(),parent_source_sha256=verify(),raw_ssd_writes=False))
if __name__=='__main__':main()
