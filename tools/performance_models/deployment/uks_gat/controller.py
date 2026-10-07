import model_context
import argparse,contextlib,fcntl,os,subprocess,time
from candidates.uks_freq_bfs_retry_v1.common import *
import json
from gpu_select import reserve
CONTROL=Path(__file__).resolve().parent
JOBS=[('smoke_gids','gids','smoke'),('smoke_digit','digit','smoke')]+[('round%d_%s'%(r,k),k,'performance') for r in range(1,6) for k in ('gids','digit')]
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    output=args.output.resolve();require(output.parent==(OUT/'runs').resolve(),'Invalid retry output')
    run(output)

def run(OUT):
    require(os.geteuid()==0,'Root required');check_ready();OUT.mkdir(exist_ok=True,parents=True)
    with contextlib.ExitStack() as stack:
        for path in [OUT/'controller.lock',Path('/tmp/digit-pa-sage-experiment.lock'),Path('/tmp/digit-pa-bidir-controller.lock'),Path('/tmp/digit-pa-sage-libnvm0.lock'),ROOT/'ssd_state/libnvm0.prepare.lock',Path('/run/digit-ae-selfservice/exclusive.lock')]:
            require(not path.is_symlink(),'Symlink lock');f=stack.enter_context(path.open('rb') if path.exists() else path.open('x+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (OUT/'status.json').exists(),'Preserve previous run')
        state=dict(source_sha256=verify(),complete=False,passed=False,started_unix=time.time(),completed=[]);reports={}
        try:
            state.update(stage='select_gpu');write(OUT/'status.json',state)
            gpu=stack.enter_context(reserve());write(OUT/'gpu_assignment.json',gpu)
            selected=gpu['selected'];os.environ['CUDA_VISIBLE_DEVICES']=selected['uuid']
            os.environ['DIGIT_AE_GPU_ASSIGNMENT']=json.dumps(selected)
            state['gpu_assignment']=selected
            state.update(stage='prepare_bfs');write(OUT/'status.json',state)
            with (OUT/'prepare.log').open('x') as log:
                subprocess.run([PYTHON,'-B','-m','candidates.uks_freq_bfs_v1.prepare'],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
            schedule=JOBS
            for name,variant,probe in schedule:
                state.update(stage=name,updated_unix=time.time());write(OUT/'status.json',state);folder=OUT/name;folder.mkdir()
                cmd=[PYTHON,'-I','-B','-u',str(CONTROL/'worker.py')]
                if variant:cmd+=['--arm',variant,'--mode',probe,'--output',str(folder)]
                with (folder/'worker.log').open('x') as log:r=subprocess.run(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                write(folder/'exit.json',dict(returncode=r.returncode));require(r.returncode==0,'Worker failed: '+name)
                report=read(folder/'report.json');model_context.checked(report);require(report['passed'] and report['source_sha256']==verify() and report['gpu_assignment']==selected and report['ae_transport_sha256']==sha(CONTROL/'transport_manifest.json'),'Invalid report');report.update(normal_exit=True,report_sha256=sha(folder/'report.json'));write(folder/'accepted.json',report);reports[name]=report;state['completed'].append(name)
            measured=[reports[n] for n,_,mode in JOBS if mode=='performance']
            require(len({r['initial_model_sha256'] for r in measured})==1,'Initial model differs')
            for arm in ('gids','digit'):
                own=[r for r in measured if r['arm']==arm]
                for key in ('roots_sha256','hot_file_sha256'):require(len({r[key] for r in own})==1,'Within-arm protocol changed: '+key)
            require(all(r['updates']==320 and r['training']['reconciled'] for r in measured),'Incomplete diagnosis')
            ratios=[reports['round%d_gids'%r]['seconds']/reports['round%d_digit'%r]['seconds'] for r in range(1,6)]
            write(OUT/'summary.json',dict(passed=True,source_sha256=verify(),runs={n:dict(seconds=r.get('seconds'),cache_line_bytes=r['cache_line_bytes']) for n,r in reports.items()},performance_result=True,gpu_assignment=selected,node_sets_differ=True,comparison='GIDS RevPR old random window vs DiGiT Freq BFS window; not matched examples',paired_speedups=ratios,max_paired_speedup=max(ratios),selection='maximum of five paired GIDS/DiGiT ratios; not proof of no interference'))
            state.update(stage='complete',complete=True,passed=True,updated_unix=time.time());write(OUT/'status.json',state)
        except BaseException as e:state.update(stage='failed',error=repr(e),updated_unix=time.time());write(OUT/'status.json',state);raise
if __name__=='__main__':main()
