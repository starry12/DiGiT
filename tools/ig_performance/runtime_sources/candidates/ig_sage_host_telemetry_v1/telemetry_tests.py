"""CPU fixtures for sampled scheduling, gaps, identity and timing alignment."""
import copy,json,tempfile,unittest
from pathlib import Path
from .telemetry import cpu_snapshot,stat_fields,cpulist,write,GPU_UUID
from .telemetry_review import review,coverage


def stat(cpu=2):
    fields=['0']*37;fields[0]='S';fields[11]='100';fields[12]='20';fields[19]='123';fields[36]=str(cpu)
    return '7 (worker (test)) '+' '.join(fields)


def fixture(folder):
    write(folder/'ready.json',dict(passed=True,candidate_sha256='fixture',target_pid=7,monitor_cpu=0,excluded_cpus=[2,58],gpu_uuid=GPU_UUID,schedstats_enabled=False))
    cpu=[];gpu=[]
    for i in range(25):
        t=int(i*.25e9);thread=dict(starttime_ticks=123,last_cpu=2,utime_ticks=i*2,stime_ticks=i,runtime_ns=i*100000,
            voluntary_switches=i*2,involuntary_switches=i,runqueue_wait_ns=None,timeslices=None)
        cpu.append(dict(ok=True,sequence=i,start_perf_ns=t,end_perf_ns=t+1000,threads={'7':thread},cpu_frequency_khz={'2':2200000},
            schedstats_enabled=False,thread_read_errors=[],cpu2_smt_siblings=[2,58],cpu_ticks={'2':[i*10,0,0,i*10,0,0,0,0],'58':[i*4,0,0,i*16,0,0,0,0]}))
    for i in range(13):
        t=int(i*.5e9);gpu.append(dict(ok=True,sequence=i,start_perf_ns=t,end_perf_ns=t+1000,sm_mhz=2400,memory_mhz=9000,graphics_mhz=2400,
            pstate=0,throttle_reasons=1,power_mw=150000,temperature_c=50))
    marker=dict(phase='training',index=0,start_perf_ns=1000000000,end_perf_ns=4000000000,seconds=3.)
    def lines(name,values):(folder/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
    lines('cpu.jsonl',cpu);lines('gpu.jsonl',gpu);lines('windows.jsonl',[marker])
    write(folder/'summary.json',dict(candidate_sha256='fixture',target_pid=7,monitor_cpu=0,samples=len(cpu),complete=True,stopped_by_worker=True,gpu_returncode=0,cpu_seconds=.02,wall_seconds=6.))
    write(folder/'exit.json',dict(returncode=0));write(folder/'gpu_exit.json',dict(cpu_seconds=.01))
    r=dict(candidate_sha256='fixture',resource_observations=dict(worker_pid=7),warmup=None,training=dict(windows=[dict(index=0,seconds=3.)]))
    return r,cpu,gpu,lines


class Checks(unittest.TestCase):
    def test_proc_parser_and_disabled_wait_is_unavailable(self):
        self.assertEqual(cpulist('0-2,58'),[0,1,2,58]);self.assertEqual(stat_fields(stat())['last_cpu'],2)
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);proc=base/'proc';sys=base/'cpu';task=proc/'7/task/7';task.mkdir(parents=True)
            (task/'stat').write_text(stat());(task/'status').write_text('voluntary_ctxt_switches: 5\nnonvoluntary_ctxt_switches: 3\nCpus_allowed_list: 2\n')
            (task/'schedstat').write_text('1000 999 8\n');(proc/'sys/kernel').mkdir(parents=True)
            setting=proc/'sys/kernel/sched_schedstats';setting.write_text('0\n');(proc/'stat').write_text('cpu2 1 2 3 4 5 6 7 8\n')
            (sys/'cpu2/topology').mkdir(parents=True);(sys/'cpu2/topology/thread_siblings_list').write_text('2,58\n')
            for c in (2,58):(sys/('cpu%d/cpufreq'%c)).mkdir(parents=True);(sys/('cpu%d/cpufreq/scaling_cur_freq'%c)).write_text('2200000\n')
            a=cpu_snapshot(7,proc,sys);self.assertTrue(a['ok']);self.assertIsNone(a['threads']['7']['runqueue_wait_ns'])
            setting.write_text('1\n');self.assertEqual(cpu_snapshot(7,proc,sys)['threads']['7']['runqueue_wait_ns'],999)

    def test_missing_samples_are_not_healthy_idle(self):
        rows=[dict(ok=False,start_perf_ns=100,end_perf_ns=200)]
        _,status=coverage(rows,0,300,1.);self.assertFalse(status['complete']);self.assertEqual(status['samples'],0)

    def test_complete_alignment_switches_and_smt(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);r,*_=fixture(folder);v=review(folder,r)
            self.assertTrue(v['diagnostic_complete']);w=v['windows'][0]
            self.assertFalse(w['scheduling']['runqueue_wait_available'])
            self.assertIsNone(w['scheduling']['main_thread']['runqueue_wait_ns'])
            self.assertEqual(w['scheduling']['cpu2_and_smt_utilization']['2']['busy_percent'],50.)
            self.assertEqual(w['scheduling']['cpu2_and_smt_utilization']['58']['busy_percent'],20.)
            self.assertGreater(w['scheduling']['main_thread']['involuntary_switches'],0)
            self.assertEqual(w['main_thread_cpu_frequency_khz']['median'],2200000)
            self.assertEqual(w['gpu_throttle_reason_masks'],[1])

    def test_gpu_gap_is_diagnostic_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);r,cpu,gpu,lines=fixture(folder)
            for row in gpu:row['ok']=False;row['error']='fixture timeout'
            lines('gpu.jsonl',gpu);v=review(folder,r)
            self.assertFalse(v['diagnostic_complete']);self.assertTrue(v['windows'][0]['gpu_query_errors'])

    def test_wrong_worker_version_or_window_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);r,*_=fixture(folder)
            for key in ('pid','version','duration'):
                bad=copy.deepcopy(r)
                if key=='pid':bad['resource_observations']['worker_pid']=8
                if key=='version':bad['candidate_sha256']='old'
                if key=='duration':bad['training']['windows'][0]['seconds']=4.
                with self.assertRaises(RuntimeError):review(folder,bad)

    def test_counter_reset_and_lost_bracket_remain_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);r,cpu,gpu,lines=fixture(folder)
            for row in cpu:
                if row['start_perf_ns']>=4e9:row['threads']['7']['runtime_ns']=0
            lines('cpu.jsonl',cpu);v=review(folder,r)
            self.assertFalse(v['diagnostic_complete']);self.assertFalse(v['windows'][0]['scheduling']['available'])


if __name__=='__main__':unittest.main(verbosity=2)
