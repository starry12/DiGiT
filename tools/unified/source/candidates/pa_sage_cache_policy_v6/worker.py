"""Controller children only; fresh process per smoke and full arm."""
import argparse
import os
from pathlib import Path
import time
from .common import read,require,sha,write,heavy_gate,verify,lock,progress,complete_handshake


def train(a,p,execution):
    from .binding import check
    from .prepared import check as check_prepared
    binding=read(a.binding);check(binding,a.protocol,execution)
    prepared=check_prepared(read(a.prepared),p,a.protocol,a.binding,execution)
    if a.mode=='full':
        from .validation import smoke_matrix
        smoke_matrix(read(a.short_receipt),p,execution,sha(a.protocol),sha(a.prepared))
    from .admission import estimate,live
    admission=live(estimate(p,binding['manifest'],a.arm));require(admission['passed'],'Native resource admission failed')
    write(a.output/'admission.json',admission)
    from ae.common import check_device
    check_device()
    from .runtime import prepare_imports,loader_kwargs
    from .backend import install_on_loader
    from .common import binary_receipt
    cls,module=prepare_imports()
    import torch,numpy as np
    from candidates.pa_sage_cache_policy_v3.native_support import startup,graph_sampler
    from .observations import CheckpointObservations
    startup();observer=CheckpointObservations(a.output/'resources.json');observer.mark('setup')
    started=time.perf_counter();progress(a.output,'loading_graph',arm=a.arm)
    graph,arrays,bundle,sampler=graph_sampler(p,binding,p['seed']);observer.mark('graph_ready')
    kwargs=loader_kwargs(p,a.arm,binding['manifest'],bundle.io_geometry)
    progress(a.output,'allocating_native_cache',arm=a.arm);loader=cls(**kwargs)
    hotdesc=prepared['arms'][a.arm];require(sha(hotdesc['path'])==hotdesc['sha256'],'Hot set checksum changed')
    hot=np.load(hotdesc['path'],mmap_mode='r')
    observer.mark('cache_allocated');progress(a.output,'cpu_preload',arm=a.arm)
    cache=install_on_loader(loader,module,p['arms'][a.arm],hot,bundle.arrays['node_to_primary_row'],bundle.arrays['storage_to_node'])
    observer.mark('cpu_preloaded');setup_seconds=time.perf_counter()-started
    require(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0] <= admission['required_bytes'], 'Actual setup exceeds admitted envelope')
    from .training import run
    report=run(p,a.arm,a.mode,loader,graph,bundle,sampler,binding,a.output,observer)
    check(binding,a.protocol,execution);check_prepared(prepared,p,a.protocol,a.binding,execution)
    require(verify()==execution,'Sources changed during training')
    report.update(source_sha256=execution,protocol_sha256=sha(a.protocol),binding_sha256=sha(a.binding),
        prepared_sha256=sha(a.prepared),native_short_receipt_sha256=sha(a.short_receipt) if a.mode=='full' else None,
        graph_sha256=binding['graph_sha256'],layout_sha256=binding['layout_sha256'],
        feature_receipt_sha256=binding['feature_receipt_sha256'],hot_nodes_sha256=hotdesc['sha256'],
        backend_binary_sha256=binary_receipt()['binary_sha256'],cache_installation=cache,admission=admission,
        resource_observations=observer.result(),setup_seconds=setup_seconds,raw_ssd_writes=False)
    complete_handshake(a.output,report) # retain resources until report validation and controller release


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','full'),required=True)
    for name in ('protocol','binding','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('prepared','short-receipt'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--arm',choices=('degree','revpr','freq','digit'))
    parser.add_argument('--controller-pid',type=int,required=True)
    a=parser.parse_args()
    from candidates.pa_sage_cache_policy_v3.supervision import bind_parent
    bind_parent(a.controller_pid)
    heavy_gate()
    require(os.getppid()==a.controller_pid,'Worker must be a direct child of the owning controller')
    execution=verify();p=read(a.protocol)
    from .protocol import validate
    validate(p);a.output.mkdir(parents=True,exist_ok=False)
    require(a.prepared is not None and a.arm is not None,'Missing prepared inputs/arm')
    with lock('/tmp/digit-pa-sage-libnvm0.lock'):
        train(a,p,execution)


if __name__=='__main__':main()
