"""Full-epoch derivative of the accepted UKS worker; tail retained and roots checked."""
import argparse,os,time,sys
from pathlib import Path
import numpy as np
from candidates.uks_freq_bfs_retry_v1.common import *
from .common import window_ranges
from .monitor import snapshot as host_snapshot
from candidates.uks_native_v1.binding import protocol,check
from candidates.uks_freq_bfs_retry_v1.common import select_sampler as graph_sampler

def admission():
    from ae.common import host
    import subprocess
    q=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=30).strip().split(',')
    require(q[0].strip()=='GPU-927ce617-743a-4bfe-6a60-8a8311cfc703' and int(q[1])*2**20>=40*2**30,'GPU2 not free enough')
    require(host()>=256*2**30,'Need 256 GiB available for initial UKS native admission')

def run(variant,folder,profile_mode):
    arm='gids' if variant=='gids_default' else 'digit'
    from candidates.uks_mixed_1k2k_v1.runtime import prepare_imports,loader_kwargs,arm_budget
    cls,module=prepare_imports()
    import torch,dgl
    from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP
    from digit.io_geometry import IOGeometry
    from candidates.uks_mixed_1k2k_v1.backend import install_on_loader
    from candidates.uks_mixed_1k2k_v1.counters import snapshot,interval
    from candidates.uks_native_v1.model import create
    from candidates.uks_freq_bfs_retry_v1.common import load_plan,check_ready
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
        hot=np.load(hot_path(arm),mmap_mode='r');primary=np.arange(p['nodes'],dtype=np.int64);storage=primary
    else:
        hot=np.load(hot_path(arm),mmap_mode='r');primary=artifact.arrays['node_to_primary_row'];storage=artifact.arrays['storage_to_node']
    installed=install_on_loader(loader,module,arm_budget(p,arm),hot,primary,storage)
    net,opt,initial=create(p,device='cuda');labels=np.load(DATA/'synthetic/labels.npy',mmap_mode='r')
    roots=load_roots(arm)
    require(len(roots)==p['cpu_cache_rows'],'Full training split required')
    roots=roots[:320*1024].copy()
    count=(len(roots)+1023)//1024
    require(count==320,'Exact 20+300 prefix required')
    initial_roots=digest(roots)
    from candidates.uks_revpr_diagnostic_v1.timing import StageTimer
    timer=StageTimer(torch.cuda.synchronize,enabled=profile_mode=='stages')
    def batches():
        for i in range(count):
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
    measured_before=None;epoch_wall_start=time.perf_counter()
    for lo,hi in window_ranges(count):
        before=snapshot(loader.BAM_FS)
        if lo==20:measured_before=before
        pending=[];seen_roots=[];rows_seen=0;timer.reset()
        host_before=host_snapshot()
        torch.cuda.synchronize();started=time.perf_counter()
        for i in range(lo,hi):
            if profile_mode=='stages':
                batch=next(it)
                with timer.stage('feature_fetch'):
                    inp,out,blocks,x=loader.fetch_feature(256,iter([batch]),torch.device('cuda:0'))
            else:
                inp,out,blocks,x=loader.fetch_feature(256,it,torch.device('cuda:0'))
            expected=roots[i*1024:(i+1)*1024]
            require(len(out)==len(expected),'Wrong root batch size')
            seen_roots.append(out.detach())
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
        host_after=host_snapshot()
        after=snapshot(loader.BAM_FS);region=interval(before,after,rows_seen,complete_region=False)
        actual=torch.cat(seen_roots).cpu().numpy()
        require(np.array_equal(actual,roots[lo*1024:min(hi*1024,len(roots))]),'Delivered root order differs')
        losses=torch.stack(pending).cpu().numpy();require(np.isfinite(losses).all(),'Nonfinite losses')
        loss_values.extend(map(float,losses));all_rows+=rows_seen
        if lo>=20:measured_rows+=rows_seen
        window=dict(phase='warmup' if lo==0 else 'training',batch_start=lo,batch_end=hi,batches=hi-lo,seconds=seconds,region=region,stages=timer.report(),host_before=host_before,host_after=host_after)
        windows.append(window);write(folder/'windows.json',windows)
        write(folder/'progress.json',dict(stage=window['phase'],arm=arm,updates=hi,total_updates=count,measured_updates=max(0,hi-20)))
    epoch_wall_seconds=time.perf_counter()-epoch_wall_start
    final_snapshot=snapshot(loader.BAM_FS)
    whole=interval(cold,final_snapshot,all_rows,complete_region=True)
    training=interval(measured_before,final_snapshot,measured_rows,complete_region=False)
    require(all(torch.isfinite(v).all().item() for v in net.parameters()),'Nonfinite final model')
    from candidates.pa_sage_cache_policy_v1.training import model_hash
    final=model_hash(net);require(final!=initial,'Optimizer did not update model')
    if arm=='digit':require(sampler._cuda_call_counter==count,'Native sampler update mismatch')
    aff=dict(initial=affinity,final=affinity_snapshot());validate_affinity(aff,'digit_full' if variant=='digit_cpu2' else 'gids')
    return dict(passed=True,arm=arm,variant=variant,profile_mode=profile_mode,hot_policy='revpr' if arm=='gids' else 'freq100_seed23',bfs_enabled=arm=='digit',hot_file_sha256=sha(hot_path(arm)),order_policy='old_random_window' if arm=='gids' else 'full_training_induced_bfs_block_shuffle_window',updates=count,measured_batches=count-20,warmup_batches=20,examples=len(roots),tail_size=len(roots)%1024 or 1024,root_coverage_exact=True,
        roots_sha256=initial_roots,initial_model_sha256=initial,final_model_sha256=final,losses=loss_values,shapes=shapes,
        windows=windows,windows_seconds=[w['seconds'] for w in windows[1:]],seconds=sum(w['seconds'] for w in windows),post_warmup_seconds=sum(w['seconds'] for w in windows[1:]),
        warmup_seconds=windows[0]['seconds'],epoch_wall_seconds=epoch_wall_seconds,setup_seconds=setup_seconds,training=training,complete_region=whole,cache=installed,affinity=aff,native=True)
