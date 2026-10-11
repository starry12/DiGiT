"""Bounded CPU-only checks of release files, input headers and import resolution."""
from ae.common import *

def check(dataset,arm,imports=False):
    import numpy as np
    execution=verify_release()
    data=ROOT/'data'/dataset
    folder=data/('bundle' if dataset=='papers' else 'full')
    manifest=read(folder/'manifest.json')
    checked=[]
    for key,desc in manifest['files'].items():
        path=folder/desc['path']; require(path.is_file(),'Missing array: '+str(path))
        value=np.load(path,mmap_mode='r',allow_pickle=False)
        require(list(value.shape)==desc['shape'],'Wrong shape: '+key)
        require(value.dtype.str==desc['dtype'],'Wrong dtype: '+key)
        expected=value.offset+value.size*value.dtype.itemsize
        require(path.stat().st_size==expected,'Wrong file length: '+key)
        checked.append(dict(name=key,shape=list(value.shape),bytes=expected))
        del value
    cfg=read(ROOT/'configs'/dataset/('protocol.json' if dataset=='papers' else 'dataset.json'))
    for name in (('source_features','label_identity') if dataset=='papers' else ('feature','labels')):
        entry=cfg['dataset'][name];require(Path(entry['path']).is_file(),'Missing source '+entry['path'])
    if dataset=='papers':
        locations=digit_paths.locations()
        for name in ('papers_csc','papers_test_trace','papers_splits','papers_baseline'):
            require(Path(locations[name]).is_dir(),'Missing external input: '+locations[name])
        for name in ('gids_cpu_rows.npy','full_cpu_rows.npy','original_degrees_i32.npy'):require((data/name).is_file(),'Missing '+name)
        from digit.eval_trace import EvaluationTrace
        expected=read(ROOT/'configs/papers/inputs_ready.json')['traces']
        for name,path in [('valid',data/'valid_trace'),('test',Path(locations['papers_test_trace']))]:
            trace=EvaluationTrace(path)
            require(trace.manifest_sha256==expected[name]['manifest_sha256'],'Trace manifest changed: '+name)
            # Verify every trace shard exists; content hashes are checked by its loader.
            for desc in trace.manifest.get('batches',[]):
                if isinstance(desc,dict) and 'path' in desc:require((path/desc['path']).exists(),'Missing trace shard')
    else:
        cache=read(ROOT/'configs/igb/gids_cache.json')
        require(Path(cache['path']).stat().st_size==cache['bytes'],'Wrong GIDS cache size')
        require(sha(cache['rows_binding']['path'])==cache['rows_binding']['sha256'],'Wrong CPU cache rows')
        for name in ('original_indptr.npy','original_indices.npy','original_eids.npy'):
            require((data/'csc'/name).is_file(),'Missing normalized CSC: '+name)
    payload=check_payload(dataset+'_'+('full' if arm=='digit_full' else 'gids'))
    modules={}
    if imports:
        import importlib
        names=['runner','model_adapter','digit.DiGiTSamplerCUDA','digit.DiGiTOutputCUDA','BAM_Feature_Store'] if dataset=='papers' else ['engine','native_fetch','IGGroupWarpCUDA','IGSageNative']
        for name in names:
            mod=importlib.import_module(name);modules[name]=str(Path(mod.__file__).resolve())
            require(str(ROOT/'ae') in modules[name],'Module resolved outside extracted runtime: '+name)
        if dataset=='papers':
            import runner
            runner.startup()  # CPU thread/RNG setup; no CUDA allocation.
    return dict(passed=True,dataset=dataset,arm=arm,release_sha256=execution,arrays=checked,imports=modules,
                ssd_state_checked=True,source_hashes_checked=True,full_data_hashes_checked=False,
                gpu_allocations=False,training_run=False,raw_ssd_io=False)
