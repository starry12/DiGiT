"""Reuse the accepted initialization for continuous mixed-read BFS training."""
import argparse
import fcntl
import math
import random
import resource
from .common import *
from candidates.pa_sage_512b_pair_v4.payload import validate_payload
from candidates.pa_sage_cpu_affinity_v1 import affinity as cpu_affinity
from candidates.pa_sage_512b_pair_v1.runtime_helpers import install_cache, verify_digit_blocks
from .graph_io import load_graph, load_bundle, overlay_manifest
from .admission import estimate as admission, stage_check


def snapshot(loader):
    return dict(feature=loader.get_feature_access_stats(), gpu=loader.get_gpu_cache_stats(),
                device=loader.get_device_io_stats())


def run(folder, smoke, order_mode):
    read_mode="mixed"
    mean_mode="legacy"
    group_mode = "incremental"
    dense_mode = "graph"
    variant = "overlap"
    arm = 'digit'
    changes = dict(frontier=True,functional=True)
    p = read(HERE / 'protocol.json')
    execution = verify()
    binding = read(OUT / 'inputs.json')
    check_inputs(binding)
    payload_extent=validate_payload(binding,p,arm)
    require(os.geteuid() == 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == '2',
            'This native candidate requires administrator execution on physical GPU 2')
    from ae.common import check_device, check_payload, host
    require(host() >= p['host_admission_gib'] * 2**30, 'Host memory admission failed')
    check_device()
    require(check_payload('papers_gids') == binding['ssd_payload'], 'SSD payload receipt changed')
    setup()
    import numpy as np
    import torch
    import dgl
    torch.set_num_threads(p['cpu_threads'])
    import BAM_Feature_Store
    from GIDS import GIDS_DGLDataLoader, get_sample_record, get_fetch_feature_record
    from candidates.pa_sage_512b_pair_v4.loader import BoundedGIDS
    from .loader import MultiEpochGIDS as OverlapGIDS
    from candidates.pa_sage_512b_overlap_v1.loader import batches as scheduled_batches
    LegacyAlignedGIDS = BoundedGIDS if variant=='legacy' else OverlapGIDS
    from candidates.pa_sage_legacy_align_v1.graph import verify_blocks
    from digit.sampler import DiGiTNeighborSampler, DIGIT_STORAGE_ROW
    from models import SAGE
    from ae.pa_sage.common import splits
    require(Path(__import__('sys').modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve() == BINARY,
            'Wrong native binary imported')
    require(getattr(BAM_Feature_Store,'FIFO_WINDOW_512B_API',0)==1,'Wrong FIFO/window backend')
    require(getattr(BAM_Feature_Store,'ASYNC_FEATURE_READ_API',0)==1,'Wrong async backend')
    require(Path(__import__('sys').modules['GIDS.GIDS'].__file__).resolve()==ROOT/'candidates/pa_sage_512b_pair_v3/runtime/GIDS/GIDS.py','Wrong GIDS adapter')
    if changes['frontier']:
        from candidates.pa_sage_group_incremental_v1.sampler import IncrementalGroupSampler
        DiGiTNeighborSampler = IncrementalGroupSampler
    import DiGiTGroupIncrementalCUDA
    require(Path(DiGiTGroupIncrementalCUDA.__file__).resolve()==SAMPLER_BINARY,"Wrong group extension")
    require(DiGiTGroupIncrementalCUDA.INCREMENTAL_GROUP_API==1,"Wrong incremental group API")
    if changes['functional']:
        from candidates.pa_sage_mean_fused_v1.model import MeanGraphSAGE, prepare_model
        SAGE = MeanGraphSAGE
    import DiGiTMeanNormalizeCUDA
    require(Path(DiGiTMeanNormalizeCUDA.__file__).resolve()==NORMALIZATION_BINARY, "Wrong normalization binary")
    require(DiGiTMeanNormalizeCUDA.MEAN_NORMALIZE_API==1, "Wrong normalization API")
    plan=admission(p,overlay_manifest(binding),arm)
    free,total=torch.cuda.mem_get_info()
    require(free>=plan['required_bytes'], 'GPU memory admission failed: free=%d required=%d' % (free,plan['required_bytes']))
    folder.mkdir(parents=True, exist_ok=False)
    write(folder/'admission.json',dict(plan,free_bytes=free,total_bytes=total,passed=True))
    setup_start = time.perf_counter()
    progress(folder, 'loading_bidirectional_graph', smoke=smoke)
    graph, arrays = load_graph(binding, p['nodes'], p['edges'])
    bundle=load_bundle(binding) if arm=='digit' else None
    sampler=(DiGiTNeighborSampler(p['fanouts'],bundle,cuda_mode='required',
                metadata_mode=p['digit_metadata_mode'],random_seed=p['seed'],group_mode=group_mode) if bundle else
                dgl.dataloading.NeighborSampler(p['fanouts'],replace=False))
    if bundle:
        stage_check(folder,plan,'before_metadata',('metadata','native','cpu_map','labels','dense'))
        progress(folder,'uploading_compact_sampler_metadata')
        sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    stage_check(folder,plan,'before_native_cache',('native','cpu_map','labels','dense'))
    progress(folder, 'initializing_512b_cache')
    payload=binding['digit_payload'] if bundle else binding['ssd_payload']
    storage_rows=bundle.manifest['feature']['num_storage_rows'] if bundle else p['nodes']
    require(storage_rows==payload_extent['runtime_rows'],'Runtime extent differs from validated payload')
    loader = LegacyAlignedGIDS(page_size=512, off=payload['offset'], cache_dim=128,
        num_ele=storage_rows*128, num_ssd=1, ssd_list=[0], cache_size=p['cache_mib'], ctrl_idx=0,
        window_buffer=True, wb_size=2, accumulator_flag=True, feature_index_mode=p['feature_index_modes'][arm],
        gpu_cache_policy=p['gpu_policies'][arm], cpu_feature_path='mapped', mixed_io=False, device_io_stats=True)
    loader.pair_enabled=(read_mode=='mixed')
    validate_layout(bundle)
    require(getattr(BAM_Feature_Store,'PA_PAIR_ASYNC_API',0)==1,'Missing paired backend')
    loader.set_required_storage_access(**p['accumulator_parameters'])
    require(math.isclose(loader.required_accesses,p['expected_required_accesses']),'Accumulator threshold changed')
    stage_check(folder,plan,'before_cpu_map',('cpu_map','labels','dense'))
    progress(folder,'preloading_cpu_cache',arm=arm,rows=p['cpu_rows'])
    cpu_cache,cache_refs=install_cache(loader,arm,binding,bundle,p)
    # Pin no full feature matrix. CPU feature pages are preloaded by BaM above.
    stage_check(folder,plan,'before_labels',('labels','dense'))
    labels_np = np.load(binding['labels'], mmap_mode='r').reshape(-1)
    labels = torch.from_numpy(np.array(labels_np, dtype=np.int64)).cuda()
    train = np.sort(splits('train'))
    require(hashlib.sha256(train.tobytes()).hexdigest() == binding['train_sorted_sha256'], 'Train split changed')
    roots = (np.random.default_rng(0).permutation(train)[:p['smoke_examples']] if smoke else train)
    features = np.load(binding['features'], mmap_mode='r')
    random.seed(p['seed']); np.random.seed(p['seed']); torch.manual_seed(p['seed'])
    torch.cuda.manual_seed_all(p['seed']); dgl.seed(p['seed']); dgl.random.seed(p['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SAGE(128, p['hidden'], p['classes'], p['layers'], p['dropout'], mean_mode=mean_mode).cuda()
    initial=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        initial.update(name.encode());initial.update(tensor.detach().cpu().numpy().tobytes())
    initial_model_sha256=initial.hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), **p['optimizer'])
    model.train()
    progress(folder, 'preparing_last_dense', dense_mode=dense_mode)
    preparation=prepare_model(model,optimizer,dense_mode,batch_size=p['batch_size'])
    require(preparation['retained_allocation_bytes']<=256*2**20,'Dense pool exceeded admission allowance')
    write(folder/'dense_preparation.json',preparation)
    stage_check(folder,plan,'before_training',())
    from .training import train_and_evaluate
    report=train_and_evaluate(folder,smoke,order_mode,p,graph,arrays,bundle,sampler,loader,model,
        optimizer,train,labels,features,binding,initial_model_sha256,cpu_cache,payload_extent,setup_start)
    write(folder/'report.json',report)
    validate_completion(report)
    check_inputs(binding)
    require(verify()==execution,'Candidate changed during training')
    write(folder/'accepted.json',dict(passed=True,report_sha256=sha(folder/'report.json'),
        candidate_sha256=execution,binary_sha256=sha(BINARY)))
    progress(folder,'complete',order_mode=order_mode,epochs=len(report['epochs']),training_seconds=report['training_seconds'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--variant',choices=VARIANTS[1:],required=True)
    parser.add_argument('--lock-fd',type=int,required=True)
    args=parser.parse_args()
    inherited=os.fstat(args.lock_fd);current=Path('/tmp/digit-pa-sage-libnvm0.lock').stat()
    require((inherited.st_dev,inherited.st_ino)==(current.st_dev,current.st_ino),'Wrong SSD lock')
    fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run(args.output,args.smoke,args.variant)


if __name__=='__main__':main()
