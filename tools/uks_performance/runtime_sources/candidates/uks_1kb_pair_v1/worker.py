"""Bounded UKS/SAGE performance worker; no feature writes or accuracy evaluation."""
import argparse,os,time,sys
from pathlib import Path
import numpy as np
from .common import *
from candidates.uks_native_v1.binding import protocol,check
from .common import select_sampler as graph_sampler

def admission():
    from ae.common import host
    import subprocess
    q=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=30).strip().split(',')
    require(q[0].strip()=='GPU-927ce617-743a-4bfe-6a60-8a8311cfc703' and int(q[1])*2**20>=40*2**30,'GPU2 not free enough')
    require(host()>=256*2**30,'Need 256 GiB available for initial UKS native admission')

def run(variant,folder,profile_mode):
    arm='gids' if variant=='gids_default' else 'digit'
    from .runtime import prepare_imports,loader_kwargs,arm_budget
    cls,module=prepare_imports()
    import torch,dgl
    from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP
    from digit.io_geometry import IOGeometry
    from .backend import install_on_loader
    from .counters import snapshot,interval
    from candidates.uks_native_v1.model import create
    from .common import load_plan,check_ready
    from candidates.ig_sage_affinity_pair_v1.affinity import apply_affinity,snapshot as affinity_snapshot,validate_affinity
    affinity=apply_affinity('digit_full' if variant=='digit_cpu2' else 'gids')
    began_setup=time.perf_counter()
    p=protocol();b=check();graph,arrays,artifact,sampler=graph_sampler(arm,0)
    geometry=IOGeometry.create(feature_row_bytes=1024,group_size=1,minimum_transfer_bytes=1024,target_request_bytes=1024)
    rows=p['nodes'] if arm=='gids' else b['receipts']['g2']['num_storage_rows']
    from candidates.uks_native_v1.storage import plan,api,state_path
    v=load_plan(arm);api()._validate_active_state(state_path(v),v,'verified')
    from candidates.uks_native_retry_v1.receipts import validate_saved_receipt
    verified=validate_saved_receipt(native_parent.OUT/('storage_'+arm)/api().VERIFY_RECEIPT,v)
    pool=dict(passed=True,feature_mode='logical_synthetic',row_bytes=1024,offset=v.device_offset_bytes,verified_bytes=v.payload_bytes)
    loader=cls(**loader_kwargs(p,arm,rows,pool,geometry))
    if arm=='gids':
        hot=np.load(DATA/'rank/hot_nodes.npy',mmap_mode='r');primary=np.arange(p['nodes'],dtype=np.int64);storage=primary
    else:
        hot=np.load(DATA/'rank/hot_nodes.npy',mmap_mode='r');primary=artifact.arrays['node_to_primary_row'];storage=artifact.arrays['storage_to_node']
    installed=install_on_loader(loader,module,arm_budget(p,arm),hot,primary,storage)
    net,opt,initial=create(p,device='cuda');labels=np.load(DATA/'synthetic/labels.npy',mmap_mode='r')
    roots=np.load(DATA/'synthetic/roots.npy',mmap_mode='r')[:320*1024].copy()
    require(len(roots)==320*1024,'Not enough complete root batches')
    initial_roots=digest(roots)
    from .timing import StageTimer
    timer=StageTimer(torch.cuda.synchronize,enabled=profile_mode=='stages')
    def batches():
        for i in range(320):
            with timer.stage('root_transfer'):
                root=torch.from_numpy(roots[i*1024:(i+1)*1024].copy()).cuda()
            with timer.stage('sampling'):
                inp,out,blocks=sampler.sample_blocks(graph,root)
            with timer.stage('address_annotation'):
                if arm=='gids':
                    blocks[0].srcdata[DIGIT_STORAGE_ROW]=inp
                    blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP]=torch.zeros_like(inp,dtype=torch.bool)
            yield inp,out,blocks
    it=iter(batches());loader.BAM_FS.begin_useful_io_region();cold=snapshot(loader.BAM_FS)
    setup_seconds=time.perf_counter()-began_setup;windows=[];all_rows=0;measured_rows=0;shapes=[];loss_values=[]
    measured_before=None
    for lo,hi in ((0,20),(20,120),(120,220),(220,320)):
        before=snapshot(loader.BAM_FS)
        if lo==20:measured_before=before
        pending=[];rows_seen=0;timer.reset()
        torch.cuda.synchronize();started=time.perf_counter()
        for i in range(lo,hi):
            if profile_mode=='stages':
                batch=next(it)
                with timer.stage('feature_fetch'):
                    inp,out,blocks,x=loader.fetch_feature(256,iter([batch]),torch.device('cuda:0'))
            else:
                inp,out,blocks,x=loader.fetch_feature(256,it,torch.device('cuda:0'))
            require(len(out)==1024,'Incomplete root batch')
            with timer.stage('label_transfer'):
                target=torch.from_numpy(np.array(labels[roots[i*1024:(i+1)*1024]],copy=True)).cuda()
            with timer.stage('forward'):
                pred=net(blocks,x)
            with timer.stage('loss'):
                loss=torch.nn.functional.cross_entropy(pred,target)
            with timer.stage('zero_grad'):
                opt.zero_grad(set_to_none=True)
            with timer.stage('backward'):
                loss.backward()
            with timer.stage('adam'):
                opt.step()
            pending.append(loss.detach());rows_seen+=len(inp)
            shapes.append([len(inp),len(out),*[b.num_edges() for b in blocks]])
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        after=snapshot(loader.BAM_FS);region=interval(before,after,rows_seen,complete_region=False)
        losses=torch.stack(pending).cpu().numpy();require(np.isfinite(losses).all(),'Nonfinite losses')
        loss_values.extend(map(float,losses));all_rows+=rows_seen
        if lo>=20:measured_rows+=rows_seen
        window=dict(phase='warmup' if lo==0 else 'training',batch_start=lo,batch_end=hi,batches=hi-lo,seconds=seconds,region=region,stages=timer.report())
        windows.append(window);write(folder/'windows.json',windows)
        write(folder/'progress.json',dict(stage=window['phase'],arm=arm,updates=hi,total_updates=320,measured_updates=max(0,hi-20)))
    final_snapshot=snapshot(loader.BAM_FS)
    whole=interval(cold,final_snapshot,all_rows,complete_region=True)
    training=interval(measured_before,final_snapshot,measured_rows,complete_region=False)
    require(all(torch.isfinite(v).all().item() for v in net.parameters()),'Nonfinite final model')
    from candidates.pa_sage_cache_policy_v1.training import model_hash
    final=model_hash(net);require(final!=initial,'Optimizer did not update model')
    if arm=='digit':require(sampler._cuda_call_counter==320,'Native sampler update mismatch')
    aff=dict(initial=affinity,final=affinity_snapshot());validate_affinity(aff,'digit_full' if variant=='digit_cpu2' else 'gids')
    return dict(passed=True,arm=arm,variant=variant,profile_mode=profile_mode,hot_policy='same_revpr',hot_file_sha256=sha(DATA/'rank/hot_nodes.npy'),updates=320,measured_batches=300,warmup_batches=20,examples=327680,
        roots_sha256=initial_roots,initial_model_sha256=initial,final_model_sha256=final,losses=loss_values,shapes=shapes,
        windows=windows,windows_seconds=[w['seconds'] for w in windows[1:]],seconds=sum(w['seconds'] for w in windows[1:]),
        warmup_seconds=windows[0]['seconds'],setup_seconds=setup_seconds,training=training,complete_region=whole,cache=installed,affinity=aff,native=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--arm',choices=['gids','digit'],required=True);a.add_argument('--mode',choices=['smoke','performance'],required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root GPU2 required');check_ready();admission()
    from ae.common import check_device
    check_device();args.output.mkdir(exist_ok=True,parents=True)
    if args.mode=='smoke':
        from .smoke import smoke
        result=smoke(args.arm,args.output)
    else:result=run('gids_default' if args.arm=='gids' else 'digit_cpu2',args.output,'off')
    check_ready();write(args.output/'report.json',dict(result,source_sha256=verify(),cache_line_bytes=1024,raw_ssd_writes=False))
if __name__=='__main__':main()
