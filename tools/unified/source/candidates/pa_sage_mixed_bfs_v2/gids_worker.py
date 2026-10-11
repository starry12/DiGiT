"""A fresh native process, either tail-sensitive smoke or full official epoch."""
import argparse
import fcntl
import math
import random
import resource
from .common import *
from candidates.pa_sage_512b_pair_v4.payload import validate_payload
from candidates.pa_sage_cpu_affinity_v1 import affinity as cpu_affinity
from candidates.pa_sage_512b_pair_v1.runtime_helpers import load_graph, load_bundle, admission, install_cache, verify_digit_blocks


def snapshot(loader):
    return dict(feature=loader.get_feature_access_stats(), gpu=loader.get_gpu_cache_stats(),
                device=loader.get_device_io_stats())


def run(folder, smoke):
    arm="gids"
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
    gids_reference.setup()
    import numpy as np
    import torch
    import dgl
    torch.set_num_threads(p['cpu_threads'])
    import BAM_Feature_Store
    from GIDS import GIDS_DGLDataLoader, get_sample_record, get_fetch_feature_record
    from .loader import GIDSEpochLoader as LegacyAlignedGIDS
    from candidates.pa_sage_legacy_align_v1.graph import verify_blocks
    from digit.sampler import DiGiTNeighborSampler, DIGIT_STORAGE_ROW
    from models import SAGE
    from ae.pa_sage.common import splits
    require(Path(__import__('sys').modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve() == GIDS_BINARY,
            'Wrong native binary imported')
    require(getattr(BAM_Feature_Store,'FIFO_WINDOW_512B_API',0)==1,'Wrong FIFO/window backend')
    require(Path(__import__('sys').modules['GIDS.GIDS'].__file__).resolve()==ROOT/'candidates/pa_sage_512b_pair_v3/runtime/GIDS/GIDS.py','Wrong GIDS adapter')
    plan=admission(p,read(Path(binding['bundle'])/'manifest.json'),arm)
    plan['components_bytes']['evaluation_transient_allowance']=512*2**20
    plan['required_bytes']+=math.ceil(512*2**20*1.15)
    free,total=torch.cuda.mem_get_info()
    require(free>=plan['required_bytes'], 'GPU memory admission failed: free=%d required=%d' % (free,plan['required_bytes']))
    folder.mkdir(parents=True, exist_ok=False)
    write(folder/'admission.json',dict(plan,free_bytes=free,total_bytes=total,passed=True))
    setup_start = time.perf_counter()
    progress(folder, 'loading_directed_graph', smoke=smoke)
    graph, arrays = load_graph(binding, p['nodes'], p['edges'])
    bundle=load_bundle(binding) if arm=='digit' else None
    sampler=(DiGiTNeighborSampler(p['fanouts'],bundle,cuda_mode='required',
                metadata_mode=p['digit_metadata_mode'],random_seed=p['seed']) if bundle else
                dgl.dataloading.NeighborSampler(p['fanouts'],replace=False))
    if bundle:
        progress(folder,'uploading_compact_sampler_metadata')
        sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    progress(folder, 'initializing_512b_cache')
    payload=binding['digit_payload'] if bundle else binding['ssd_payload']
    storage_rows=bundle.manifest['feature']['num_storage_rows'] if bundle else p['nodes']
    require(storage_rows==payload_extent['runtime_rows'],'Runtime extent differs from validated payload')
    loader = LegacyAlignedGIDS(page_size=512, off=payload['offset'], cache_dim=128,
        num_ele=storage_rows*128, num_ssd=1, ssd_list=[0], cache_size=p['cache_mib'], ctrl_idx=0,
        window_buffer=True, wb_size=2, accumulator_flag=True, feature_index_mode=p['feature_index_modes'][arm],
        gpu_cache_policy=p['gpu_policies'][arm], cpu_feature_path='mapped', mixed_io=False, device_io_stats=True)
    loader.set_required_storage_access(**p['accumulator_parameters'])
    require(math.isclose(loader.required_accesses,p['expected_required_accesses']),'Accumulator threshold changed')
    progress(folder,'preloading_cpu_cache',arm=arm,rows=p['cpu_rows'])
    cpu_cache,cache_refs=install_cache(loader,arm,binding,bundle,p)
    # Pin no full feature matrix. CPU feature pages are preloaded by BaM above.
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
    model = SAGE(128, p['hidden'], p['classes'], p['layers'], p['dropout']).cuda()
    initial=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        initial.update(name.encode());initial.update(tensor.detach().cpu().numpy().tobytes())
    initial_model_sha256=initial.hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), **p['optimizer'])
    model.train()
    from .training import train_and_evaluate
    report=train_and_evaluate(folder,smoke,'gids_decay',p,graph,arrays,bundle,sampler,loader,model,
        optimizer,train,labels,features,binding,initial_model_sha256,cpu_cache,payload_extent,setup_start)
    write(folder/'report.json',report);validate_completion(report);check_inputs(binding)
    require(verify()==execution,'Candidate changed during training')
    write(folder/'accepted.json',dict(passed=True,report_sha256=sha(folder/'report.json'),
        candidate_sha256=execution,binary_sha256=sha(GIDS_BINARY)))
    progress(folder,'complete',variant='gids_decay',epochs=len(report['epochs']),training_seconds=report['training_seconds'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--lock-fd',type=int,required=True)
    args=parser.parse_args();inherited=os.fstat(args.lock_fd);current=Path('/tmp/digit-pa-sage-libnvm0.lock').stat()
    require((inherited.st_dev,inherited.st_ino)==(current.st_dev,current.st_ino),'Wrong SSD lock')
    fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run(args.output,args.smoke)


if __name__=='__main__':main()
