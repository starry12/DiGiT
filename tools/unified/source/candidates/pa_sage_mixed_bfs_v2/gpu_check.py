"""Twenty-epoch synchronous/async lifecycles using CUDA gathers, no SSD loading."""
import argparse
from types import SimpleNamespace
from .common import *


def run(output):
    setup();source=verify()
    import torch
    from .loader import MultiEpochGIDS,GIDSEpochLoader
    from candidates.pa_sage_512b_overlap_v1.gpu_tests import FakeAsyncStore,_FixtureDataLoader
    from candidates.pa_sage_512b_overlap_v1.loader import batches
    class Store(FakeAsyncStore):
        def read_feature_pair_async(self,count,outputs,indices,rows,flags,tables,capacities,stream):
            require(len(flags)==len(tables)==len(capacities)==count,'Missing pair pointer owners')
            return self.read_feature_merged_async(count,outputs,indices,rows,128,128,stream)
    data=torch.arange(512*128,dtype=torch.float32,device='cuda:0').reshape(512,128)
    store=Store(data);loader=MultiEpochGIDS.__new__(MultiEpochGIDS)
    loader.BAM_FS=store;loader.gids_device='cuda:0';loader.cache_dim=128
    loader.wb_size=2;loader.required_accesses=6832.4;loader.pair_enabled=True
    loader.accumulator_flag=loader.window_buffering_flag=True;loader.heterograph=False
    loader.feature_index_mode='explicit';loader.graph_reorganize=False;loader.feature_map=None
    loader.strict_feature_row_validation=True
    loader.feature_index_batches={k:0 for k in ('logical','explicit','legacy_map')}
    loader.feature_index_rows=dict(loader.feature_index_batches)
    stream=None
    for epoch in range(20):
        loader.begin_epoch()
        if stream is None:stream=loader.io_stream.cuda_stream
        require(loader.io_stream.cuda_stream==stream,'Replaced native stream between epochs')
        roots=[(torch.arange(n,device='cuda:0')+epoch)%512 for n in (64,64,64,7)]
        source_batches=[(r,r,[SimpleNamespace(srcdata={'digit_storage_row':r,'_ID':r,
                         'digit_storage_is_group':torch.zeros(len(r),dtype=torch.bool,device='cuda:0')})]) for r in roots]
        seen=[]
        for inp,out,blocks,x in batches(_FixtureDataLoader(source_batches,loader),loader,'overlap'):
            require(torch.equal(x,data.index_select(0,inp)),'Wrong async features after epoch reset')
            seen.append(len(out))
        require(seen==[64,64,64,7] and loader.pipeline.summary()['drained'],'Lost epoch/tail')
        require(loader.live_ticket is None and not loader.hint_refs and loader.submitted_reads==loader.collected_reads,'Pending epoch pointers')
    loader.live_ticket=object()
    try:loader.begin_epoch()
    except RuntimeError:pass
    else:raise RuntimeError('Accepted epoch reset with live native ticket')
    loader.live_ticket=None
    baseline=GIDSEpochLoader.__new__(GIDSEpochLoader)
    baseline.BAM_FS=FakeAsyncStore(data)
    for key in ('gids_device','cache_dim','wb_size','required_accesses','accumulator_flag',
                'window_buffering_flag','heterograph','graph_reorganize','feature_map',
                'strict_feature_row_validation'):
        setattr(baseline,key,getattr(loader,key))
    baseline.feature_index_mode='logical'
    baseline.feature_index_batches={k:0 for k in ('logical','explicit','legacy_map')}
    baseline.feature_index_rows=dict(baseline.feature_index_batches)
    native_owner=baseline.BAM_FS
    for epoch in range(20):
        baseline.begin_epoch()
        roots=[(torch.arange(n,device='cuda:0')+epoch)%512 for n in (64,64,64,7)]
        source_batches=[(r,r,[]) for r in roots]
        seen=[]
        for inp,out,blocks,x in batches(_FixtureDataLoader(source_batches,baseline),baseline,'legacy'):
            require(torch.equal(x,data.index_select(0,inp)),'Wrong synchronous features after epoch reset')
            seen.append(len(out))
        require(seen==[64,64,64,7] and baseline.pipeline.summary()['drained'],'Lost baseline epoch/tail')
        require(baseline.BAM_FS is native_owner,'Replaced baseline native owner')
    baseline.pipeline=SimpleNamespace(summary=lambda:dict(drained=False))
    try:baseline.begin_epoch()
    except RuntimeError:pass
    else:raise RuntimeError('Accepted baseline epoch reset before draining')
    torch.cuda.synchronize()
    write(output,dict(passed=True,candidate_sha256=source,binary_sha256=sha(BINARY),
        epochs=20,updates=80,gids_epochs=20,gids_updates=80,
        one_native_owner=True,gids_one_native_owner=True,features_exact=True,raw_ssd_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
