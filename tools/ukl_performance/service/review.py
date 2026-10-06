"""Independent ten-worker acceptance; only same-round ratios are selectable."""
import math
from common import *
def review(experiment,source_sha):
 experiment=Path(experiment);series=read(experiment/'latest_status.json')
 require(series.get('passed') is True and series.get('state')=='complete'
         and series.get('rounds_complete')==5 and series.get('runs_complete')==10,'Five complete pairs required')
 require(series.get('warmup_batches')==20 and series.get('measured_batches')==300,'Window mismatch')
 require(len(series['pairs'])==5 and len(series['runs'])==10,'Incomplete pair records')
 seen=set();ratios=[];timings=[]
 for n,pair in enumerate(series['pairs'],1):
  require(pair['round']==n and pair['pair_id']==series['run_id']+'_r'+str(n),'Pair identity mismatch')
  seconds={}
  for arm in ('gids','digit'):
   ref=pair[arm];p=Path(ref['path'])
   require(p.name=='acceptance.json' and p.parent.parent==experiment and p.resolve()==p and p not in seen,'Invalid or repeated acceptance')
   seen.add(p);require(sha(p)==ref['sha256'],'Acceptance hash changed');a=read(p);w=a['worker'];s=a['worker_state']
   require(a['passed'] is True and a['stage']==arm and a['pair_id']==pair['pair_id'] and a['selected_gpu']==series['selected_gpu'],'Arm identity/acceptance mismatch')
   require(all(a[k] is True for k in ('kernel_monitor_ok','post_guard_passed','gpu_released','io_preflight_passed')),'Exit checks failed')
   require(s['Result']=='success' and s['ExecMainStatus']=='0' and s['MainPID']=='0','Worker exit failed')
   require(a['manifest_sha256']==source_sha and w['manifest_sha256']==source_sha,'Source version mismatch')
   require(w['passed'] is True and w['updates']==320 and w['warmup_batches']==20 and w['measured_batches']==300 and w['finite'] is True,'Training not accepted')
   require(w['graph_and_aux_released'] and w['external_cache_released'] and not w['raw_ssd_writes'],'Ownership/write mismatch')
   seconds[arm]=w['seconds'];require(type(seconds[arm]) in (int,float) and math.isfinite(seconds[arm]) and seconds[arm]>0 and seconds[arm]==ref['seconds'],'Invalid timing')
   require(w['sampler_variant']==('gids_gpu_common_v26' if arm=='gids' else 'gpu'),'Sampler mismatch')
   require(w['execution_optimizations']==dict(reused_sampling_buffers=True,gpu_postprocessing=True) and w['timing_protocol']['stage_profiling'] is False,'Execution protocol mismatch')
   require(w['cpu_affinity']['policy']==('default' if arm=='gids' else 'cpu2'),'Affinity mismatch')
   if arm=='digit':
    require(w['cpu_affinity']['cpus']==[2] and w['cpu_affinity_initial']['cpus']==[2],'CPU2 not inherited')
    m=w['numa_placement'];require(m['policy']=='interleave:0-1' and len(m['regions'])==3 and {r['label'] for r in m['regions']}=={'graph','auxiliary','cpu_features'},'NUMA regions missing')
    for r in m['regions']:
     counts=r['node_sample_pages'];total=r['sampled_pages']
     require(r['verified'] is True and r['before_first_touch'] is True and r['migration'] is False and r['nodes']==[0,1] and r['policy']=='interleave','NUMA policy mismatch')
     require(set(counts)=={'0','1'} and total==4096 and sum(counts.values())==total and all(2*total<=5*v<=3*total for v in counts.values()),'NUMA placement unbalanced')
   else:require(w['numa_placement']==dict(policy='unchanged',regions=[]),'GIDS placement changed')
   require(ref==series['runs'][(n-1)*2+(arm=='digit')],'Runs/pairs disagree')
  ratio=seconds['gids']/seconds['digit'];require(ratio==pair['speedup'],'Pair ratio mismatch');ratios.append(ratio);timings.append(seconds)
 best=max(range(5),key=lambda i:ratios[i]);require(series['speedup']==ratios[best] and series['best_round']==best+1,'Wrong maximum paired ratio')
 return dict(passed=True,formal_workers=10,rounds=5,speedup=ratios[best],best_round=best+1,
             selection='maximum same-round GIDS / DiGiT speedup across five accepted pairs',
             gpu=series['selected_gpu'],source_sha256=source_sha,series_sha256=sha(experiment/'latest_status.json'),
             native_ae_replay_accepted=True)
