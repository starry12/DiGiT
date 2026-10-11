import argparse,json,os,subprocess,time
from pathlib import Path
from .common import ROOT,OUT,read,write,sha,verify

REFERENCE=OUT/'gpu/stage_gids_run_20261011_211229_1678215'
PY='/home/embed/miniconda3/envs/gids/bin/python'

def reference():
    a=read(REFERENCE/'acceptance.json');r=read(REFERENCE/'common_training.json')
    if a['passed'] or a['worker_state']['ExecMainStatus']!='1':
        raise RuntimeError('Historical reference state changed')
    if not (r['complete'] and r['updates']==320 and r['arm']=='gids'):
        raise RuntimeError('Historical reference computation incomplete')
    return dict(accepted=False,reference_only=True,seconds=r['measured_seconds'],
        common_training_path=str(REFERENCE/'common_training.json'),
        common_training_sha256=sha(REFERENCE/'common_training.json'),
        acceptance_path=str(REFERENCE/'acceptance.json'),acceptance_sha256=sha(REFERENCE/'acceptance.json'),
        source_sha256=a['manifest_sha256'],worker_exit_code=1)

def summarize(entry,source,prior):
    path=Path(entry['path']);a=read(path)
    if sha(path)!=entry['sha256']:raise RuntimeError('Acceptance changed')
    from .runtime import protocol as P
    if not (entry['arm']=='digit' and a['stage']=='digit' and a['passed']
            and a['execution_scope']=='single_arm_diagnostic' and a['manifest_sha256']==source
            and a['pair_id']==entry['pair_id'] and a['pair_id'].endswith('_single_digit')
            and a['kernel_monitor_ok'] and a['post_guard_passed'] and a['gpu_released']
            and a['io_preflight_passed'] and a['source_verification']['passed']
            and P.worker_passed(a['worker_state'],a['worker'],'digit')):
        raise RuntimeError('Standalone DiGiT receipt rejected')
    return dict(schema='ukl-single-arm-v1',dataset='UKL',mode='short1',arm='digit',
        passed=True,complete=True,execution_scope='single_arm_diagnostic',
        paired_comparison_complete=False,migration_acceptance_passed=False,
        source_sha256=source,acceptance=entry,common_training=a['worker']['common_training'],
        historical_gids_reference=prior)

def control(folder):
    if os.geteuid()!=0:raise RuntimeError('Use the root service')
    os.environ.update(UKL_UNIFIED_MODE='short1',UKL_UNIFIED_REPETITION='1',UKL_UNIFIED_SINGLE_ARM='digit')
    from candidates.group_cache_migration_v1.common import stable
    stable('UKL')
    try:
        source=verify();prior=reference()
        from candidates.unified_experiments_v2.run import wait_resources
        wait_resources(folder)
        write(folder/'status.json',dict(stage='preflight',complete=False,passed=False))
        for module in ('candidates.large_graph_fast_v1.gpu_check','candidates.ukl_compact_cover_v1.gpu_check'):
            subprocess.run([PY,'-B','-m',module],check=True,cwd=ROOT)
        from .runtime import start
        if (start.P.SCOPE,start.P.REPETITION,start.P.SINGLE_ARM)!=('short1',1,'digit'):
            raise RuntimeError('Standalone runtime configuration mismatch')
        write(folder/'status.json',dict(stage='running_digit',complete=False,passed=False,backend=str(start.P.OUT)))
        entry=start.execute()
        if verify()!=source:raise RuntimeError('Source changed')
        result=summarize(entry,source,prior)
        write(folder/'summary.json',result)
        write(folder/'status.json',dict(stage='complete',complete=True,passed=True,paired_comparison_complete=False))
        return result
    except BaseException as e:
        write(folder/'status.json',dict(stage='failed',complete=False,passed=False,error=repr(e)));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--control',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.control:
        if a.output is None:raise ValueError('Output required')
        control(a.output);return
    from candidates.group_cache_migration_v1.common import stable
    stable('UKL');source=verify()
    from candidates.unified_experiments_v2.registry import plan
    prepared=plan('UKL','short1')
    if prepared['missing_preparation']:raise RuntimeError(str(prepared['missing_preparation']))
    stamp=time.strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    folder=OUT/'single_runs'/stamp;unit='digit-ukl-single-'+stamp
    cmd=['/usr/bin/systemd-run','--unit='+unit,'--service-type=exec',
        '--property=WorkingDirectory='+str(ROOT),'--property=Restart=no',
        '--property=KillMode=control-group','--property=LimitMEMLOCK=infinity',
        '--property=MemorySwapMax=0']
    env=dict(CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(ROOT/'bam/build/lib'),
        UKL_UNIFIED_MODE='short1',UKL_UNIFIED_REPETITION='1',UKL_UNIFIED_SINGLE_ARM='digit')
    cmd+=['--setenv='+k+'='+v for k,v in env.items()]
    cmd += [PY,'-B','-u','-m','candidates.ukl_unified_v2.single','--control','--output',str(folder)]
    receipt=dict(source_sha256=source,output=str(folder),unit=unit,command=cmd,
        execution_scope='single_arm_diagnostic',historical_gids_reference=reference())
    if a.execute:
        if os.geteuid()!=0:raise RuntimeError('Use sudo for the bounded service')
        folder.mkdir(parents=True,exist_ok=False);write(folder/'plan.json',receipt)
        subprocess.run(cmd,check=True);write(OUT/'latest_single_digit.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
