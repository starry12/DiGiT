"""Fresh native process for a smoke or one complete seed-0 system run."""
import argparse,fcntl,os
import numpy as np
from candidates.pa_sage_layout_shared_resume_v3.common import *

def identity(path):
    s=Path(path).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def check_inputs(binding):
    for path,item in binding['files'].items():require(identity(ROOT/path)==item['identity'],'Input changed: '+path)

def execute(a):
    execution=verify();p=cfg();policy=p['arms'][a.arm];binding=read(a.binding);check_inputs(binding)
    require(binding['candidate_sha256']==execution and binding['protocol_sha256']==sha(P),'Wrong input binding')
    from ae.common import check_device,check_payload
    check_device();a.output.mkdir(parents=True,exist_ok=False)
    setup();import torch,dgl,runner as r
    from digit import DiGiTSamplerCUDA
    import BAM_Feature_Store
    require(Path(DiGiTSamplerCUDA.__file__).resolve()==PARENT/'runtime/digit/DiGiTSamplerCUDA.so','Wrong sampler binary')
    require(Path(__import__('sys').modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()==ROOT/'candidates/io_accounting_v1/runtime/BAM_Feature_Store/BAM_Feature_Store.so','Wrong I/O binary')
    from candidates.io_accounting_v1.accounting import install_runner,summarize
    install_runner(r)
    from candidates.pa_sage_layout_shared_resume_v3 import training_loop as loop
    from candidates.pa_sage_layout_shared_resume_v3.admission import estimate
    from candidates.pa_sage_bidir_native_v2.overlay import apply
    from digit.artifacts import load_artifact_bundle
    from digit.gpu_admission import check_live
    from GIDS import GIDS
    from ae.pa_sage.observations import CheckpointObservations
    observer=CheckpointObservations(a.output/'resources.json')
    def mark(stage,**kw):observer.mark(stage);progress(a.output,stage,arm=a.arm,smoke=a.smoke,**kw)
    r.startup();plan=check_live(estimate());write(a.output/'admission.json',plan)
    require(plan['passed'],'Live GPU/host admission failed; see admission.json')
    require(plan['free_bytes']>plan['total_bytes']-2**30,'GPU 2 is already in use')
    mark('start');start=time.perf_counter();data=ROOT/p['data'];base=ROOT/p['base_layout'];overlay=Path(p['overlay']);prepared=read(overlay/'overlay_receipt.json')
    require(prepared['passed'] and sha(overlay/'overlay_receipt.json')==binding['prepared_sha256'],'Unverified graph overlay')
    from .pool import check_receipt,source_rows
    full=check_receipt(p['feature_mode'])
    from .bundle import load_metadata
    basebundle=load_metadata(base)
    bundle=apply(basebundle,np.load(overlay/'reordered_indptr.npy',mmap_mode='r'),np.load(overlay/'reordered_indices.npy',mmap_mode='r'),p['graph']['edges']) if basebundle else None
    payload=full;rows=basebundle.manifest['feature']['num_storage_rows']
    require(rows*512<=full['verified_bytes'],'Layout exceeds fully verified shared extent')
    mark('native_cache_initializing')
    from candidates.pa_sage_layout_shared_resume_v3.settings import loader_kwargs
    loader=GIDS(**loader_kwargs(p,basebundle.manifest,payload['offset'],basebundle.io_geometry))
    mark('native_cache_allocated')
    cpu_rows=np.load(base/'full_cpu_rows.npy',mmap_mode='r') if policy['cpu_cache_rows'] else np.empty(0,dtype=np.int64)
    require(len(cpu_rows)==policy['cpu_cache_rows'],'CPU budget differs')
    if len(cpu_rows):
        loader.cpu_backing_buffer(128,len(cpu_rows));loader.set_cpu_buffer(torch.from_numpy(cpu_rows.copy()),len(cpu_rows))
    torch.cuda.synchronize();loader.reset_device_io_stats();mark('cpu_preloaded')
    from candidates.pa_sage_bidir_native_v2.graph_io import load_pinned_csc
    g,arrays=load_pinned_csc(data,p['graph']['nodes'],p['graph']['edges'])
    mark('graph_loaded',graph_mapping='private_copy_on_write')
    degree=np.diff(arrays[0]);require(degree.max()<=2**31-1,'Degree overflow');degrees=torch.from_numpy(degree.astype(np.int32)).cuda();del degree
    labels=np.load(source_config()['label_identity']['path'],mmap_mode='r').reshape(-1);mark('degrees_loaded')
    def progress_hook(**kw):
        stage=kw.pop('stage');observer.mark(stage+'_epoch_'+str(kw.get('epoch',0)));progress(a.output,stage,**kw)
    loop.progress=progress_hook
    protocol=dict(paper_specified=dict(fanouts=p['fanouts'],batch_size=p['batch_size'],epochs=p['epochs'],group_size=p['group_size']))
    features=(np.load(source_config()['source_features']['path'],mmap_mode='r') if p['feature_mode']=='real' else source_rows('shared')) if a.smoke else None
    setup_seconds=time.perf_counter()-start
    result=loop.run_training(a.arm,0,loader,bundle,g,labels,degrees,a.output/'report.json',data,protocol,smoke=a.smoke,features=features)
    mark('training_complete')
    require(verify()==execution,'Candidate changed during training');check_inputs(binding)
    check_receipt(p['feature_mode'])
    result.update(candidate_sha256=execution,protocol_sha256=sha(P),input_binding_sha256=sha(a.binding),prepared_sha256=sha(overlay/'overlay_receipt.json'),
                  graph_mapping='private_copy_on_write',graph=p['graph'],metadata_mode=p['metadata_mode'],raw_ssd_writes=False,source_only=False,seed=0,
                  point=p['point'],layout_manifest_sha256=sha(base/'final/bundle/manifest.json'),overlay_receipt_sha256=sha(overlay/'overlay_receipt.json'),policy=policy,cpu_cache_rows=len(cpu_rows),cache_bytes=p['gpu_cache_bytes'],admission=plan,resource_observations=observer.result(),
                  feature_mode=p['feature_mode'],pool_receipt_sha256=sha(full['receipt_path']),pool_offset=full['offset'],
                  feature_semantics='logical_node_real' if p['feature_mode']=='real' else 'physical_row_proxy_not_accuracy',
                  verification_policy=p['verification_policy'],
                  setup_seconds=setup_seconds,worker_seconds=time.perf_counter()-start,
                  io_accounting_training=summarize([(e['training'],e['train_seconds']) for e in result['epochs']]),
                  io_accounting_validation=None,io_accounting_test=None)
    write(a.output/'report.json',result);write(a.output/'worker_ready.json',dict(passed=True,report_sha256=sha(a.output/'report.json')))
    deadline=time.monotonic()+120
    while not (a.output/'release_worker.json').exists():
        require(time.monotonic()<deadline,'Monitor shutdown handshake timed out');time.sleep(.25)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('digit_full',),required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True);a=parser.parse_args()
    require(not a.smoke or cfg()['verification_policy']['independent_native_smoke'],'No independent smoke for non-pilot configurations')
    require(__debug__ and os.geteuid()==0,'Native entry needs local sudo, without Python optimization')
    with open('/tmp/digit-pa-sage-libnvm0.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__':main()
