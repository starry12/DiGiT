"""Fresh native process for a smoke or one complete seed-0 system run."""
import argparse,fcntl,os
import numpy as np
from training.gcn.common import *

def identity(path):
    s=Path(path).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def check_inputs(binding):
    for path,item in binding['files'].items():require(identity(ROOT/path)==item['identity'],'Input changed: '+path)

def execute(a):
    execution=verify();p=cfg();binding=read(a.binding);check_inputs(binding)
    require(binding['candidate_sha256']==execution and binding['protocol_sha256']==sha(P),'Wrong input binding')
    from ae.common import check_device,check_payload
    check_device();a.output.mkdir(parents=True,exist_ok=False)
    setup();import torch,dgl,runner as r
    from digit import DiGiTSamplerCUDA
    import BAM_Feature_Store
    require(Path(DiGiTSamplerCUDA.__file__).resolve()==PARENT/'runtime/digit/DiGiTSamplerCUDA.so','Wrong sampler binary')
    require(Path(__import__('sys').modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()==ROOT/'runtime/io/runtime/BAM_Feature_Store/BAM_Feature_Store.so','Wrong I/O binary')
    from runtime.io.accounting import install_runner,summarize
    install_runner(r)
    from training.gcn import training_loop as loop
    from training.gcn.admission import estimate
    from training.sage.overlay import apply
    from digit.artifacts import load_artifact_bundle
    from digit.gpu_admission import check_live
    from GIDS import GIDS
    from ae.pa_sage.observations import CheckpointObservations
    observer=CheckpointObservations(a.output/'resources.json')
    def mark(stage,**kw):observer.mark(stage);progress(a.output,stage,arm=a.arm,smoke=a.smoke,**kw)
    r.startup();plan=check_live(estimate());require(plan['passed'],'Live GPU/host admission failed')
    require(plan['free_bytes']>plan['total_bytes']-2**30,'GPU 2 is already in use')
    mark('start');start=time.perf_counter();data=ROOT/p['data'];base=ROOT/p['base_layout'];prepared=read(data/'prepared.json')
    require(prepared['passed'] and sha(data/'prepared.json')==binding['prepared_sha256'],'Unverified graph overlay')
    plain=check_payload('papers_gids');full=read(base/'ssd_ready.json')
    require(full['full_readback_passed'] and sha(full['state'])==full['state_sha256'],'SSD payload receipt changed')
    require(sha(full['verify_receipt'])==full['verify_receipt_sha256'],'SSD readback receipt changed')
    basebundle=load_artifact_bundle(base/'final/bundle',validation_mode='fast') if a.arm=='digit_full' else None
    bundle=apply(basebundle,np.load(data/'reordered_indptr.npy',mmap_mode='r'),np.load(data/'reordered_indices.npy',mmap_mode='r'),p['graph']['edges']) if basebundle else None
    payload=full if bundle else plain;rows=basebundle.manifest['feature']['num_storage_rows'] if bundle else plain['bytes']//512
    if bundle:
        from digit.ssd_payload import validate_bundle_verify_receipt
        validate_bundle_verify_receipt(full['verify_receipt'],basebundle,device_offset_bytes=full['offset'])
    mark('native_cache_initializing')
    loader=GIDS(page_size=4096,off=payload['offset'],cache_dim=128,num_ele=rows*128,num_ssd=1,ssd_list=[0],cache_size=4096,
                ctrl_idx=0,window_buffer=False,accumulator_flag=False,feature_index_mode='explicit' if bundle else 'logical',
                gpu_cache_policy='fifo' if bundle else 'legacy',cpu_feature_path='mapped',mixed_io=bool(bundle),device_io_stats=True,
                mixed_io_geometry=basebundle.io_geometry.with_payload_offset(payload['offset']).to_native_mapping() if bundle else None)
    mark('native_cache_allocated')
    cpu_rows=np.load(base/('full_cpu_rows.npy' if bundle else 'gids_cpu_rows.npy'),mmap_mode='r')
    require(len(cpu_rows)==p['cpu_cache_rows'],'CPU budget differs');loader.cpu_backing_buffer(128,len(cpu_rows));loader.set_cpu_buffer(torch.from_numpy(cpu_rows.copy()),len(cpu_rows))
    torch.cuda.synchronize();loader.reset_device_io_stats();mark('cpu_preloaded')
    from training.sage.graph_io import load_pinned_csc
    g,arrays=load_pinned_csc(data,p['graph']['nodes'],p['graph']['edges'])
    mark('graph_loaded',graph_mapping='private_copy_on_write')
    degree=np.diff(arrays[0]);require(degree.max()<=2**31-1,'Degree overflow');degrees=torch.from_numpy(degree.astype(np.int32)).cuda();del degree
    labels=np.load(source_config()['label_identity']['path'],mmap_mode='r').reshape(-1);mark('degrees_loaded')
    def progress_hook(**kw):
        stage=kw.pop('stage');observer.mark(stage+'_epoch_'+str(kw.get('epoch',0)));progress(a.output,stage,**kw)
    loop.progress=progress_hook
    protocol=dict(paper_specified=dict(fanouts=p['fanouts'],batch_size=p['batch_size'],epochs=p['epochs'],group_size=2))
    features=np.load(source_config()['source_features']['path'],mmap_mode='r') if a.smoke else None
    setup_seconds=time.perf_counter()-start
    result=loop.run_training(a.arm,0,loader,bundle,g,labels,degrees,a.output/'report.json',data,protocol,smoke=a.smoke,features=features)
    mark('training_complete')
    require(verify()==execution,'Candidate changed during training');check_inputs(binding)
    require(sha(base/'ssd_ready.json')==binding['files'][str((base/'ssd_ready.json').relative_to(ROOT))]['sha256'],'SSD receipt changed')
    def phase_io(regions,phase):
        v=summarize(regions)
        v['phase']=phase
        if phase!='training':
            v['evaluation_seconds']=v.pop('train_seconds')
            v['ssd_useful_per_evaluation_gbps']=v.pop('ssd_useful_per_train_gbps')
        return v
    result.update(candidate_sha256=execution,protocol_sha256=sha(P),input_binding_sha256=sha(a.binding),prepared_sha256=sha(data/'prepared.json'),
                  graph_mapping='private_copy_on_write',graph=p['graph'],metadata_mode=p['metadata_mode'],raw_ssd_writes=False,source_only=False,seed=0,
                  cpu_cache_rows=len(cpu_rows),cache_bytes=p['gpu_cache_bytes'],admission=plan,resource_observations=observer.result(),
                  setup_seconds=setup_seconds,worker_seconds=time.perf_counter()-start,
                  io_accounting_training=summarize([(e['training'],e['train_seconds']) for e in result['epochs']]),
                  io_accounting_validation=phase_io([(e['validation'],e['validation']['seconds']) for e in result['epochs']], 'validation'),
                  io_accounting_test=None if a.smoke else phase_io([(result['test'],result['test']['seconds'])], 'test'))
    write(a.output/'report.json',result);write(a.output/'worker_ready.json',dict(passed=True,report_sha256=sha(a.output/'report.json')))
    deadline=time.monotonic()+120
    while not (a.output/'release_worker.json').exists():
        require(time.monotonic()<deadline,'Monitor shutdown handshake timed out');time.sleep(.25)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('gids','digit_full'),required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True);a=parser.parse_args()
    require(__debug__ and os.geteuid()==0,'Native entry needs local sudo, without Python optimization')
    with open('/tmp/digit-pa-sage-libnvm0.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__':main()
