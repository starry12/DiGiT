"""Preview by default; build and independently validate a new g2 adjacency."""
import argparse
import shutil
import subprocess
import time
from .common import *

def plan(output):
    source=verify();binding=read(ROOT/'results/uks_native_20260929_v1/binding.json')
    n=binding['receipts']['csc']['nodes'];g=binding['receipts']['g2'];output=Path(output).resolve()
    require(output!=DATA and DATA not in output.parents,'Cannot write into accepted base data')
    return dict(source_sha256=source,output=str(output),nodes=n,groups=g['num_primary_groups']+g['num_replica_groups'],
        commands=[[str(BINARY),mode,str(DATA),str(output),str(n)] for mode in ('build','validate')],
        maximum_output_bytes=8*(g['reordered_units']+n+g['num_primary_groups']+g['num_replica_groups']+1)+512,
        host_available_min_bytes=128*2**30,cpu_threads=1,group_size=2,feature_layout_changed=False,
        raw_ssd_writes=False,sampler_changed=False,requires_exclusive_experiment_locks=True)

def execute(output):
    p=plan(output);dest=Path(p['output']);require(not dest.exists(),'Preserve previous output directory')
    with exclusive():
        binding=source_binding();before={path:identity(path) for path in binding['files']}
        from ae.common import host
        require(host()>=p['host_available_min_bytes'],'Insufficient host memory for offline preparation')
        require(dest.parent.is_dir(),'Create the parent storage directory first')
        require(shutil.disk_usage(dest.parent).free>=p['maximum_output_bytes']+8*2**30,'Insufficient output disk space')
        started=time.time();write(OUT/'preparation_status.json',dict(stage='building',plan=p,started=started))
        try:
            reports=[]
            for mode,command in zip(('build','validate'),p['commands']):
                write(OUT/'preparation_status.json',dict(stage=mode,plan=p,started=started))
                # Put both logs outside the native-created destination; do not
                # precreate its directory or erase a failed attempt.
                log=OUT/(dest.name+'_'+mode+'.log')
                with log.open('x') as stream:
                    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=stream,text=True,check=True)
                reports.append(json.loads(result.stdout))
            require(reports[0]==reports[1],'Independent validation disagrees with builder counts')
            validation=reports[1];s=stable();files={}
            for name in ARRAYS:
                path=dest/(name+'.npy');old=identity(path);digest=sha(path)
                require(identity(path)==old,'Output changed during hashing')
                files[name]=dict(identity=old,sha256=digest,filesystem_uuid=s.filesystem_uuid(path))
            require(source_binding()==binding and all(identity(k)==v for k,v in before.items()),'Base inputs changed during build')
            require(verify()==p['source_sha256'],'Source changed during build')
            receipt=dict(passed=True,source_sha256=p['source_sha256'],base_binding_sha256=sha(ROOT/'results/uks_native_20260929_v1/binding.json'),
                base_files=binding['files'],validation=validation,files=files,group_size=2,
                feature_layout_changed=False,raw_ssd_writes=False,native_training_accepted=False,
                elapsed_seconds=time.time()-started,plan=p)
            write(dest/'prepared.json',receipt)
            write(OUT/'preparation_status.json',dict(stage='complete',passed=True,output=str(dest),validation=validation))
        except BaseException as e:
            write(OUT/'preparation_status.json',dict(stage='failed',passed=False,error=repr(e),plan=p));raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    print(json.dumps(plan(args.output),indent=2),flush=True)
    if args.execute:execute(args.output)

if __name__=='__main__':main()
