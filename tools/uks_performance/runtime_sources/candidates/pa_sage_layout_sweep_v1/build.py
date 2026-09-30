"""Future filesystem-only build entry. Large builds require the AE exclusive lock."""
import argparse
import contextlib
import fcntl
import shutil
import resource
from .common import *

class Context:
    def __init__(self,spec,inputs,nodes,edges,hot_rows,output,fixture=False):
        self.point=point(spec['group_size'],spec['replica_percent'])
        require(spec.get('id')==self.point['id'],'Point name and parameters disagree')
        self.inputs=inputs;self.nodes=nodes;self.edges=edges;self.hot_rows=hot_rows
        self.output=Path(output);self.contract_path=self.output/'contract.json';self.fixture=fixture
    def cfg(self):
        return dict(paper_specified=dict(group_size=self.point['group_size'],replication_ratio=self.point['replication_ratio']),
                    reconstruction_choices=dict(io_page_bytes=4096))
    def source_config(self):
        return dict(num_nodes=self.nodes,source_features=self.inputs['features'],
                    source_contract=dict(normalized_directed=True,inputs=self.inputs))
    def csc_paths(self):return tuple(Path(self.inputs[k]['path']) for k in ('indptr','indices'))
    def check_resources(self):
        if self.fixture:return
        require(host()>100*2**30,'Need at least 100 GiB available for filesystem metadata build')
        b=budget(self.nodes,self.edges,self.hot_rows,self.point['group_size'],self.point['replica_percent'])
        require(shutil.disk_usage(existing_parent(self.output.parent)).free>=b['filesystem_bytes_with_scratch_and_reserve'],'Insufficient filesystem space upper bound')

@contextlib.contextmanager
def exclusive(lock_path=Path('/run/digit-ae-selfservice/exclusive.lock')):
    # Existing lock only: never create a parallel lock if the active server path is missing.
    with lock_path.open('r+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

def build_layout(ctx):
    if ctx.fixture:return _build_layout(ctx)
    with exclusive():return _build_layout(ctx)

def _build_layout(ctx):
    import numpy as np
    from . import metadata
    require(not ctx.output.exists(),'Output already exists; do not overwrite/resume unrelated data')
    require(__debug__,'Do not disable assertions for preparation')
    if ctx.fixture:
        require(ctx.nodes<=4096 and ctx.edges<=131072,'Fixture exceeds lightweight limits')
        require(sum(Path(d['path']).stat().st_size for d in ctx.inputs.values())<=64*2**20,'Fixture inputs exceed 64 MiB')
    ctx.check_resources()
    ctx.output.mkdir(parents=True,exist_ok=False)
    progress(ctx.output,'binding_inputs')
    bindings={}
    started=time.perf_counter();cpu_started=time.process_time();stages={}
    try:
        for name,desc in ctx.inputs.items():
            path=Path(desc['path']);require(path.is_file(),'Expected a regular input file')
            before=identity(path);actual=sha(path)
            require(actual==desc['sha256'] and identity(path)==before,'Input hash/identity mismatch: '+name)
            bindings[name]=dict(path=str(path),sha256=actual,identity=before)
        ptr,idx=map(array,ctx.csc_paths());features=array(ctx.inputs['features']['path']);hot=array(ctx.inputs['hot_nodes']['path'])
        require(ptr.dtype==np.int64 and idx.dtype==np.int64 and ptr.shape==(ctx.nodes+1,) and idx.shape==(ctx.edges,),'Wrong CSC dimensions/dtype')
        require(ptr[0]==0 and int(ptr[-1])==ctx.edges and np.all(np.diff(ptr)>=0),'Invalid CSC pointers')
        require(not idx.size or (idx.min()>=0 and idx.max()<ctx.nodes),'Invalid node IDs')
        require(features.shape==(ctx.nodes,128) and features.dtype==np.float32,'Expected PA float32 128D features')
        require(hot.dtype==np.int64 and hot.shape==(ctx.hot_rows,) and len(hot)%8==0,'Wrong whole-page hot budget')
        require(not len(hot) or (hot[0]>=0 and hot[-1]<ctx.nodes and np.all(np.diff(hot)>0)),'Hot IDs must be unique sorted logical IDs')
        code=code_bindings()
        contract=dict(schema='digit-pa-layout-cell-v1',point=ctx.point,inputs=bindings,hot_rows=ctx.hot_rows,code_sha256=code,
                      seed=0,bfs_enabled=False,graph_policy='normalized directed; bidirectional overlay pending',
                      fixture=ctx.fixture,raw_ssd_access=False)
        write(ctx.contract_path,contract);save(ctx.output/'hot_nodes.npy',hot)
        write(ctx.output/'sources.json',dict(files={'original_indptr':bindings['indptr'],'original_indices':bindings['indices'],'features':bindings['features']}))
        stages['input_binding_and_contract_seconds']=time.perf_counter()-started
        for name,fn in (('metadata',metadata.make_metadata),('metadata_validation',metadata.validate_metadata),('payload_and_readback',metadata.payload)):
            stage_started=time.perf_counter();fn(ctx,ctx.output,ctx.output);stages[name+'_seconds']=time.perf_counter()-stage_started
        for desc in bindings.values():require(identity(desc['path'])==desc['identity'],'Input changed during build')
        require(code_bindings()==code,'Build code changed during preparation')
        m=read(ctx.output/'final/bundle/manifest.json');g=m['grouping'];rows=m['feature']['num_storage_rows']
        require(g['group_size']==ctx.point['group_size'] and g['replication_ratio']==ctx.point['replication_ratio'],'Manifest parameters differ')
        replica_rows=g['num_replica_groups']*g['group_size']
        require(replica_rows<=ctx.nodes*ctx.point['replica_percent']//100,'Replica row budget exceeded')
        report=dict(passed=True,point=ctx.point,fixture=ctx.fixture,manifest_sha256=sha(ctx.output/'final/bundle/manifest.json'),
                    input_binding_sha256=sha(ctx.contract_path),replica_rows_used=replica_rows,
                    requested_replica_row_budget=ctx.nodes*ctx.point['replica_percent']//100,
                    achieved_replica_fraction=replica_rows/ctx.nodes,storage_rows=rows,payload_bytes=rows*512,
                    padding_rows=rows-ctx.nodes-replica_rows,full_cpu_rows_sha256=sha(ctx.output/'full_cpu_rows.npy'),
                    preparation=dict(wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu_started,
                                     process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,stages=stages),
                    expanded_directed_adjacency_exact=True,filesystem_features_bit_exact=True,
                    bidirectional_overlay_ready=False,native_ready=False,raw_ssd_access=False)
        write(ctx.output/'build_receipt.json',report);progress(ctx.output,'filesystem_complete',native_ready=False)
        return report
    except BaseException as exc:
        progress(ctx.output,'failed',error=type(exc).__name__+': '+str(exc));raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--point',required=True)
    parser.add_argument('--execute-filesystem-build',action='store_true')
    args=parser.parse_args();p=read(args.plan)
    require(p['schema']=='digit-pa-sage-layout-sweep-plan-v1','Unknown plan')
    matches=[s for s in p['points'] if s['id']==args.point];require(len(matches)==1,'Unknown/duplicate grid point')
    s=matches[0]
    print(json.dumps(dict(point=s['id'],output=s['filesystem_destination'],execute=args.execute_filesystem_build,raw_ssd_access=False),indent=2))
    if not args.execute_filesystem_build:return
    require(os.geteuid()==0,'Large preparation must acquire the administrator-owned AE lock')
    ctx=Context(s,p['inputs'],p['dataset']['nodes'],p['dataset']['directed_edges'],p['hot_rows'],s['filesystem_destination'])
    build_layout(ctx)

if __name__=='__main__':main()
