"""Read-only training telemetry, outside the worker and outside timed CUDA paths."""
import argparse,ctypes,json,os,signal,subprocess,sys,time
from pathlib import Path

GPU_UUID='GPU-927ce617-743a-4bfe-6a60-8a8311cfc703'


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def cpulist(text):
    values=set()
    for part in text.strip().split(','):
        if not part:continue
        ends=part.split('-');values.update(range(int(ends[0]),int(ends[-1])+1))
    return sorted(values)


def stat_fields(text):
    # comm may contain spaces and parentheses; numeric fields start after the last ).
    f=text.rsplit(')',1)[1].split()
    return dict(state=f[0],utime_ticks=int(f[11]),stime_ticks=int(f[12]),
                starttime_ticks=int(f[19]),last_cpu=int(f[36]))


def cpu_snapshot(pid,proc=Path('/proc'),syscpu=Path('/sys/devices/system/cpu')):
    began=time.perf_counter_ns();threads={};errors=[]
    enabled=(proc/'sys/kernel/sched_schedstats').read_text().strip()=='1'
    for task in sorted((proc/str(pid)/'task').iterdir()):
        try:
            v=stat_fields((task/'stat').read_text());status={}
            for line in (task/'status').read_text().splitlines():
                if ':' in line:
                    k,x=line.split(':',1);status[k]=x.strip()
            v.update(voluntary_switches=int(status['voluntary_ctxt_switches']),
                     involuntary_switches=int(status['nonvoluntary_ctxt_switches']),
                     allowed_cpus=status['Cpus_allowed_list'])
            ss=[int(x) for x in (task/'schedstat').read_text().split()]
            v.update(runtime_ns=ss[0],runqueue_wait_ns=ss[1] if enabled else None,
                     timeslices=ss[2] if enabled else None)
            threads[task.name]=v
        except FileNotFoundError:errors.append(dict(tid=task.name,error='thread exited during sample'))
        except (OSError,ValueError,KeyError) as exc:errors.append(dict(tid=task.name,error=str(exc)))
    siblings=cpulist((syscpu/'cpu2/topology/thread_siblings_list').read_text())
    observed=sorted({v['last_cpu'] for v in threads.values()}|{2}|set(siblings))
    frequencies={}
    for cpu in observed:
        try:frequencies[str(cpu)]=int((syscpu/('cpu%d/cpufreq/scaling_cur_freq'%cpu)).read_text())
        except (OSError,ValueError):frequencies[str(cpu)]=None
    usage={}
    for line in (proc/'stat').read_text().splitlines():
        fields=line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            usage[fields[0][3:]]=[int(x) for x in fields[1:9]]
    return dict(ok=bool(threads) and str(pid) in threads,start_perf_ns=began,end_perf_ns=time.perf_counter_ns(),
                time_unix=time.time(),target_pid=pid,threads=threads,thread_read_errors=errors,
                cpu_frequency_khz=frequencies,cpu_ticks=usage,cpu2_smt_siblings=siblings,
                schedstats_enabled=enabled,clock_ticks_per_second=os.sysconf('SC_CLK_TCK'))


class ClockBackend:
    def __init__(self,gpu):
        from candidates.ig_monitor_v1.gpu_monitor import NvmlBackend
        self.backend=NvmlBackend(gpu)
        if self.backend.uuid!=GPU_UUID:raise RuntimeError('Wrong telemetry GPU UUID')

    def sample(self):
        b=self.backend;result={};errors={}
        specifications=[('graphics_mhz','nvmlDeviceGetClockInfo',[0],ctypes.c_uint),
                        ('sm_mhz','nvmlDeviceGetClockInfo',[1],ctypes.c_uint),
                        ('memory_mhz','nvmlDeviceGetClockInfo',[2],ctypes.c_uint),
                        ('pstate','nvmlDeviceGetPerformanceState',[],ctypes.c_uint),
                        ('throttle_reasons','nvmlDeviceGetCurrentClocksThrottleReasons',[],ctypes.c_ulonglong),
                        ('power_mw','nvmlDeviceGetPowerUsage',[],ctypes.c_uint),
                        ('temperature_c','nvmlDeviceGetTemperature',[0],ctypes.c_uint)]
        for name,function,args,type_ in specifications:
            value=type_()
            try:
                b.function(function,[ctypes.c_void_p]+[ctypes.c_uint]*len(args)+[ctypes.POINTER(type_)])(b.handle,*args,ctypes.byref(value))
                result[name]=int(value.value)
            except Exception as exc:result[name]=None;errors[name]=str(exc)
        return dict(result,physical_gpu_uuid=b.uuid,physical_gpu_index=int(b.gpu),errors=errors,
                    ok=all(result[k] is not None for k in ('graphics_mhz','sm_mhz','memory_mhz')))


def gpu_main(output,gpu,cpu):
    os.sched_setaffinity(0,{cpu});stopped=False
    def stop(*args):
        nonlocal stopped;stopped=True
    signal.signal(signal.SIGTERM,stop);backend=None;count=0;started=time.process_time_ns()
    with (output/'gpu.jsonl').open('x',buffering=1) as log:
        while not stopped and not (output/'stop').exists():
            began=time.perf_counter_ns()
            try:
                if backend is None:backend=ClockBackend(gpu)
                v=backend.sample()
            except Exception as exc:v=dict(ok=False,error=type(exc).__name__+': '+str(exc))
            v.update(start_perf_ns=began,end_perf_ns=time.perf_counter_ns(),time_unix=time.time(),pid=os.getpid(),sequence=count)
            log.write(json.dumps(v)+'\n');count+=1
            if not (output/'gpu_ready.json').exists():write(output/'gpu_ready.json',dict(v,monitor_cpu=cpu))
            remaining=.5-(time.perf_counter_ns()-began)/1e9
            if remaining>0:time.sleep(remaining)
    write(output/'gpu_exit.json',dict(complete=True,samples=count,cpu_seconds=(time.process_time_ns()-started)/1e9))


def collect(output,pid,gpu,candidate,available_cpus=None):
    siblings=cpulist(Path('/sys/devices/system/cpu/cpu2/topology/thread_siblings_list').read_text())
    choices=sorted(set(available_cpus if available_cpus is not None else os.sched_getaffinity(0))-set(siblings)-{2})
    if not choices:raise RuntimeError('No diagnostic CPU outside CPU2 and SMT siblings')
    cpu=choices[0];os.sched_setaffinity(0,{cpu})
    began=time.perf_counter_ns();cpu_began=time.process_time_ns();count=0;error=None;child=None
    initial=stat_fields(Path('/proc/%d/stat'%pid).read_text())['starttime_ticks']
    command=[sys.executable,'-B','-u','-m','candidates.ig_sage_host_telemetry_v1.telemetry','--gpu-worker','--output',str(output),'--gpu',str(gpu),'--monitor-cpu',str(cpu)]
    with (output/'gpu.log').open('x') as gpu_log,(output/'cpu.jsonl').open('x',buffering=1) as log:
        try:
            child=subprocess.Popen(command,stdout=gpu_log,stderr=subprocess.STDOUT)
            while not (output/'stop').exists():
                tick=time.perf_counter_ns()
                v=cpu_snapshot(pid);v['sequence']=count
                if v['threads'].get(str(pid),{}).get('starttime_ticks')!=initial:raise RuntimeError('Target exited or PID reused')
                log.write(json.dumps(v)+'\n');count+=1
                if not (output/'ready.json').exists() and (output/'gpu_ready.json').exists():
                    gr=json.loads((output/'gpu_ready.json').read_text())
                    if not gr['ok']:raise RuntimeError('GPU clock telemetry unavailable: '+str(gr))
                    write(output/'ready.json',dict(passed=True,pid=os.getpid(),gpu_pid=child.pid,target_pid=pid,target_starttime_ticks=initial,
                        candidate_sha256=candidate,monitor_cpu=cpu,excluded_cpus=siblings,gpu_uuid=gr['physical_gpu_uuid'],
                        cpu_period_seconds=.25,gpu_period_seconds=.5,schedstats_enabled=v['schedstats_enabled'],
                        cpu_frequency_source='sysfs scaling_cur_freq; sampled observation, not cycle-weighted frequency'))
                if child.poll() is not None:raise RuntimeError('GPU clock collector exited early')
                remaining=.25-(time.perf_counter_ns()-tick)/1e9
                if remaining>0:time.sleep(remaining)
        except BaseException as exc:error=type(exc).__name__+': '+str(exc)
        finally:
            try:
                v=cpu_snapshot(pid);v['sequence']=count;log.write(json.dumps(v)+'\n');count+=1
            except (OSError,ValueError):pass
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=1.5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=1)
            write(output/'summary.json',dict(complete=error is None,stopped_by_worker=(output/'stop').exists(),error=error,
                candidate_sha256=candidate,target_pid=pid,target_starttime_ticks=initial,monitor_cpu=cpu,samples=count,
                wall_seconds=(time.perf_counter_ns()-began)/1e9,cpu_seconds=(time.process_time_ns()-cpu_began)/1e9,
                gpu_returncode=child.returncode if child is not None else None))
    return 0 if error is None else 1


class TrainingTelemetry:
    def __init__(self,output,candidate,gpu=2,available_cpus=None):
        self.output=Path(output);self.candidate=candidate;self.gpu=gpu;self.process=None;self.log=None
        self.available_cpus=sorted(available_cpus if available_cpus is not None else os.sched_getaffinity(0))

    def __enter__(self):
        self.output.mkdir();self.log=(self.output/'collector.log').open('x')
        command=[sys.executable,'-B','-u','-m','candidates.ig_sage_host_telemetry_v1.telemetry','--collect','--output',str(self.output),
                 '--target-pid',str(os.getpid()),'--gpu',str(self.gpu),'--candidate',self.candidate,
                 '--available-cpus',','.join(map(str,self.available_cpus))]
        # The worker is already CPU2-bound; let only the external collector choose
        # a scheduling location from the controller's inherited CPU set.
        env=dict(os.environ)
        self.process=subprocess.Popen(command,stdout=self.log,stderr=subprocess.STDOUT,env=env)
        try:
            deadline=time.monotonic()+10
            while not (self.output/'ready.json').exists():
                if self.process.poll() is not None:raise RuntimeError('Telemetry failed before training; inspect collector.log')
                if time.monotonic()>deadline:raise RuntimeError('Telemetry readiness timeout before training')
                time.sleep(.05)
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def window(self,phase,index,started,seconds):
        # Called after the existing timer/counter collection, never inside a batch.
        with (self.output/'windows.jsonl').open('a') as f:
            f.write(json.dumps(dict(phase=phase,index=index,start_perf_ns=round(started*1e9),end_perf_ns=round((started+seconds)*1e9),seconds=seconds))+'\n')

    def __exit__(self,*args):
        (self.output/'stop').touch()
        if self.process is not None:
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.log is not None:self.log.close()
        write(self.output/'exit.json',dict(returncode=self.process.returncode if self.process is not None else None))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--gpu-worker',action='store_true');p.add_argument('--collect',action='store_true');p.add_argument('--target-pid',type=int)
    p.add_argument('--monitor-cpu',type=int);p.add_argument('--candidate');p.add_argument('--available-cpus');a=p.parse_args()
    if a.gpu_worker:gpu_main(a.output,a.gpu,a.monitor_cpu);return 0
    if a.collect:return collect(a.output,a.target_pid,a.gpu,a.candidate,cpulist(a.available_cpus) if a.available_cpus else None)
    p.error('Choose collector or GPU worker')


if __name__=='__main__':sys.exit(main())
