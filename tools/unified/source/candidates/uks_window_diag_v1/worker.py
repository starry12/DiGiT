"""Guarded, fresh native process per full-epoch arm."""
import argparse
import json
import os
from pathlib import Path
from .common import OUT,require,verify,inputs,write,load_roots,validate_inherited_locks
from contextlib import nullcontext

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('gids','digit'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=['off','stages'],required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--lock-fds');g.add_argument('--pause-lease',type=Path);args=p.parse_args()
    source=verify();binding=inputs();folder=args.output.resolve()
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root GPU2 required')
    require(folder.parent.parent==OUT/'runs' and not folder.exists(),'Unexpected/existing output')
    from candidates.uks_group_first_cpu2_v1.common import exclusive,check_prepared,DEFAULT_OUTPUT,prepared_hashes
    from candidates.uks_group_first_cpu2_v1.sampling import graph_sampler
    from candidates.uks_group_first_cpu2_v1.affinity import install
    from candidates.uks_group_first_v1.common import OUT as FIRST_OUT,verify as first_verify
    if args.pause_lease:
        require(args.pause_lease.resolve()==folder.parent/'ukl_pause.json','Unexpected lease')
        from .pause import validate
        validate(args.pause_lease)
    else:validate_inherited_locks(args.lock_fds)
    with nullcontext():
        gpu=json.loads((FIRST_OUT/'gpu_checks.json').read_text())
        require(gpu['passed'] and gpu['source_sha256']==first_verify(),'Group-first kernel not accepted')
        prepared=check_prepared(DEFAULT_OUTPUT);before=prepared_hashes()
        # Stable UUID checks must precede legacy by-value identity imports.
        from . import engine
        from ae.common import check_device
        engine.admission();check_device();folder.mkdir()
        write(folder/'progress.json',dict(stage='initializing',arm=args.arm))
        def sampling(arm,seed):
            require(arm==args.arm and seed==0,'Unexpected sampler')
            import torch,dgl
            torch.manual_seed(0);torch.cuda.manual_seed_all(0);dgl.seed(0);dgl.random.seed(0)
            return graph_sampler(arm,seed,'gids' if arm=='gids' else 'first')
        engine.graph_sampler=sampling;engine.load_roots=load_roots
        engine.hot_path=lambda arm:Path(binding['hot_sets'][arm]['path'])
        with install():result=engine.run('gids_default' if args.arm=='gids' else 'digit_cpu2',folder,args.profile)
        from candidates.cache_epoch_v1.core import digest_array
        require(result['roots_sha256']==digest_array(load_roots(args.arm)[:320*1024]),'Prefix root order changed')
        require(result['initial_model_sha256']==binding['initial_model_sha256'],'Initial model changed')
        require(result['complete_region']['raw_allocator_cache']['resident_pages_start']==0,'GPU not cold')
        require(result['hot_file_sha256']==binding['hot_sets'][args.arm]['sha256'],'Wrong hot set')
        require(check_prepared(DEFAULT_OUTPUT)==prepared and prepared_hashes()==before,'Prepared data changed')
        require(verify()==source,'Source changed during run');inputs()
        result.update(hot_policy='revpr' if args.arm=='gids' else 'full_epoch_freq_seed23',
            source_sha256=source,order_policy='matched_full_epoch_BFS_blocks',bfs_enabled=True,
            presampling_performed=False,raw_ssd_writes=False,full_epoch_training=False,accuracy_evaluated=False)
        result['measured_seconds']=result['post_warmup_seconds']
        result['measurement_scope']='20 warmup plus 300 timed batches; synchronized diagnostics separate'
        if args.pause_lease:validate(args.pause_lease)
        write(folder/'report.json',result)
        write(folder/'progress.json',dict(stage='complete',arm=args.arm,updates=result['updates'],passed=True))

if __name__=='__main__':main()
