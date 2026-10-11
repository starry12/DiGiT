"""Build/validate separate adjacency overlays; never touch raw SSD or features."""
import argparse
import math
import os
import shutil
import subprocess
import time
from .common import *

def npy(path, expected=None):
    import numpy as np
    p=Path(path).resolve();value=np.load(p,mmap_mode='r')
    require(value.dtype==np.dtype('<i8') and value.flags.c_contiguous,'Expected C-order int64')
    result=dict(path=str(p),offset=value.offset,length=value.nbytes,width=8,count=value.size,
                identity=identity(p),sha256=expected)
    del value
    return result

def source(dataset):
    stable(dataset)
    if dataset=='IG':
        base=ROOT/'data/igb';m=read(base/'full/manifest.json');c=read(base/'csc/manifest.json')
        specs={}
        for key,name in [('ptr','original_indptr.npy'),('idx','original_indices.npy')]:
            specs[key]=npy(base/'csc'/name,c['files'][name]['file_sha256'])
        for key,name in [('gp','reordered_indptr'),('gi','reordered_indices'),('members','group_members'),('super','supernode_to_group')]:
            specs[key]=npy(base/'full'/m['files'][name]['path'],m['files'][name]['sha256'])
        return dict(dataset=dataset,format='dense',nodes=m['dataset']['num_nodes'],edges=m['dataset']['num_edges'],groups=m['grouping']['num_groups'],specs=specs,
                    base_manifest_sha256=sha(base/'full/manifest.json'),csc_manifest_sha256=sha(base/'csc/manifest.json'))
    module=__import__('candidates.cache_epoch_v1.'+dataset.lower()+'.dataset',fromlist=['METADATA'])
    m=module.METADATA;b=module.BASE.binding();specs={}
    for key,name in [('ptr','indptr'),('idx','indices'),('gp','group_ptr'),('gi','group_ids'),('covered','covered'),('members','members')]:
        v=dict(b[name]);v['path']=str(Path(v['path']).resolve());v['width']=4 if key in ('idx','gi','members') else 8
        v['count']=v['length']//v['width'];specs[key]=v
    return dict(dataset=dataset,format='sparse',**{k:m[k] for k in ('nodes','edges','groups')},specs=specs,
                base_metadata=m,base_specs=b)

def descriptor(info,path):
    text='{} {} {} {}\n'.format(info['format'],info['nodes'],info['groups'],info['edges'])
    for k,v in info['specs'].items():
        require(not any(x.isspace() for x in v['path']),'Unsupported whitespace in path')
        text+='{} {} {} {} {}\n'.format(k,v['width'],v['offset'],v['count'],v['path'])
    Path(path).write_text(text)

def plan(dataset):
    info=source(dataset);n,g,e=info['nodes'],info['groups'],info['edges']
    maximum=8*(n+g+1)+8*info['specs']['gi']['count'] if info['format']=='dense' else 8*(n+1)+20*(e//2)
    return dict(dataset=dataset,output=str(DATA/dataset),maximum_output_bytes=maximum,
                min_host_available_bytes=128*2**30,threads=4,group_size=2,
                feature_layout_changed=False,raw_ssd_writes=False,source=info)

def check_prepared(dataset,full_hash=False):
    p=DATA/dataset;r=read(p/'prepared.json')
    from .preparation_compatibility import accepted_source
    require(r['passed'] and accepted_source(r['source_sha256']),'Overlay not accepted/current')
    require(r['base']==source(dataset),'Original graph binding changed')
    s=stable(dataset)
    for v in r['files'].values():
        now=identity(v['path']);require(now[1:]==v['identity'][1:] and s.filesystem_uuid(v['path'])==v['filesystem_uuid'],'Overlay identity changed')
        if full_hash:require(sha(v['path'])==v['sha256'],'Overlay digest changed')
    return r

def execute(dataset):
    p=plan(dataset);base=p['source'];dest=Path(p['output']);source_sha=verify()
    require(not dest.exists(),'Preserve existing output: '+str(dest))
    require(dest.parent.is_dir(),'Missing output parent: '+str(dest.parent))
    from candidates.uks_group_reuse_v2.common import exclusive
    from ae.common import host
    with exclusive():
        cg=next(v[3:] for v in Path('/proc/self/cgroup').read_text().splitlines() if v.startswith('0::'))
        group=Path('/sys/fs/cgroup')/cg.lstrip('/')
        require('digit-group-reuse-'+dataset.lower()+'-' in group.name and group.name.endswith('.service'),'Use the bounded preparation service')
        require((group/'memory.max').read_text().strip()==str(128*2**30) and (group/'memory.swap.max').read_text().strip()=='0','Preparation memory limits missing')
        require(host()>=p['min_host_available_bytes'],'Host memory admission failed')
        require(shutil.disk_usage(dest.parent).free>=p['maximum_output_bytes']+16*2**30,'Insufficient space for conservative output bound')
        folder=OUT/dataset;folder.mkdir(parents=True,exist_ok=True)
        spec=folder/'source.spec';require(not spec.exists(),'Preserve previous preparation attempt')
        s=stable(dataset)
        # Verify stable UUID and all remaining identity fields before any long read.
        for v in base['specs'].values():
            require(identity(v['path'])[1:]==v['identity'][1:],'Source identity changed')
        descriptor(base,spec);write(folder/'plan.json',p)
        started=time.time()
        def state(stage,**kw):write(folder/'preparation_status.json',dict(stage=stage,started_unix=started,updated_unix=time.time(),**kw))
        try:
            from candidates.cache_epoch_v1.io import direct_chunks
            for name,v in base['specs'].items():
                state('source_hash',array=name,bytes=v['identity'][2])
                require(v['sha256'] and len(v['sha256'])==64,'Missing accepted source hash')
                actual=identity(v['path']);seen=0;last=time.monotonic()
                expected=dict(identity=actual,bytes=actual[2],sha256=v['sha256'])
                for chunk in direct_chunks(v['path'],expected):
                    seen+=len(chunk);chunk.release()
                    if time.monotonic()-last>=30:state('source_hash',array=name,bytes=actual[2],bytes_done=seen);last=time.monotonic()
            reports=[]
            for mode in ('build','validate'):
                state(mode)
                cmd=[str(OUT/'build/reuse'),mode,str(spec),str(dest),str(p['threads'])]
                with (folder/(mode+'.log')).open('x') as log:
                    value=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=log,text=True,check=True)
                report=json.loads(value.stdout);write(folder/(mode+'.json'),report);reports.append(report)
            require(reports[0]==reports[1],'Preparation replay disagrees')
            state('hashing_and_binding');files={}
            for name in (('ptr.i64','units.i64') if base['format']=='dense' else ('ptr.i64','units.i32','covered.i64')):
                path=dest/name;old=identity(path);h=sha(path);require(identity(path)==old,'Output changed while hashing')
                files[name]=dict(path=str(path),offset=0,length=old[2],identity=old,sha256=h,filesystem_uuid=s.filesystem_uuid(path))
            require(base==source(dataset) and source_sha==verify(),'Source changed during preparation')
            v=reports[-1];v['adjacency_group_edge_fraction']=v['group_covered_edges']/base['edges']
            write(dest/'prepared.json',dict(passed=True,source_sha256=source_sha,base=base,files=files,validation=v,
                feature_layout_changed=False,raw_ssd_writes=False,native_training_accepted=False,elapsed_seconds=time.time()-started))
            state('complete',passed=True,output=str(dest),validation=v)
        except BaseException as e:state('failed',passed=False,error=repr(e));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['IG','UKL','CL']);p.add_argument('--execute',action='store_true');a=p.parse_args()
    print(json.dumps(plan(a.dataset),indent=2),flush=True)
    if a.execute:execute(a.dataset)
if __name__=='__main__':main()
