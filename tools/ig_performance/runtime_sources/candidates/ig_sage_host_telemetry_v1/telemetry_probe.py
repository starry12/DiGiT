"""Read-only clock/proc integration check; no CUDA, graph, features or SSD training."""
import json,os,time
from pathlib import Path
from .telemetry import TrainingTelemetry,write
from .telemetry_review import review
from .common import OUT


def main():
    old=sorted(os.sched_getaffinity(0));folder=OUT/'telemetry_probe_raw'
    if folder.exists():raise RuntimeError('Preserve prior probe evidence')
    phases={'warmup':dict(windows=[]),'training':dict(windows=[])}
    try:
        os.sched_setaffinity(0,{2})
        with TrainingTelemetry(folder,'telemetry-probe',available_cpus=old) as trace:
            for phase,index,duration in [('warmup',0,1.5),('training',0,2.),('training',1,2.),('training',2,2.)]:
                began=time.perf_counter()
                while time.perf_counter()-began<duration:
                    busy=time.perf_counter()+.01
                    while time.perf_counter()<busy:pass
                    time.sleep(.04)
                seconds=time.perf_counter()-began
                trace.window(phase,index,began,seconds)
                phases[phase]['windows'].append(dict(index=index,seconds=seconds))
                time.sleep(.1)
        result=review(folder,dict(candidate_sha256='telemetry-probe',resource_observations=dict(worker_pid=os.getpid()),**phases))
        assert result['diagnostic_complete'],result
        total=result['collector_cpu_seconds']+result['gpu_collector_cpu_seconds']
        share=100*total/result['collector_wall_seconds']
        assert share<5.,'Telemetry CPU use exceeds 5 percent of one core'
        assert all(w['scheduling']['main_thread']['voluntary_switches']>0 for w in result['windows'])
        write(OUT/'telemetry_probe.json',dict(passed=True,scope='CPU2 synthetic busy/sleep windows plus read-only GPU clock queries; no CUDA training or SSD access',
            telemetry=result,recorder_cpu_percent_one_core=share,training_overhead_claim=False))
        print(json.dumps(dict(passed=True,cpu_samples=result['cpu_samples'],gpu_samples=result['gpu_samples'],recorder_cpu_percent_one_core=share,schedstats_enabled=result['schedstats_enabled'],monitor_cpu=result['monitor_cpu']),indent=2))
    finally:os.sched_setaffinity(0,set(old))


if __name__=='__main__':main()
