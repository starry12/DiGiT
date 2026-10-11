"""Read-only per-service memory attribution; no argv, environment or data reads."""
import json,time
from pathlib import Path
from candidates.ukl_memory_ladder_v5.observer import Observer as Previous

def memory_groups(root):
    groups=[]
    for p in sorted(root.iterdir())[:2048]:
        if not p.is_dir():continue
        try:
            current=int((p/'memory.current').read_text())
            if current<64*2**20:continue
            stat=dict((k,int(v)) for k,v in (s.split() for s in (p/'memory.stat').read_text().splitlines()))
            groups.append(dict(name=p.name,current=current,**{k:stat.get(k) for k in
                ('anon','file','shmem','file_dirty','file_writeback','pgscan','pgsteal')},
                pressure=(p/'memory.pressure').read_text().strip()))
        except (OSError,ValueError) as e:groups.append(dict(name=p.name,error=type(e).__name__))
    return groups

class Observer(Previous):
    def sample(self,memory):
        result=super().sample(memory);processes=[]
        for v in result['processes'][:8192]:
            try:
                p=self.proc/str(v['pid']);text=(p/'status').read_text();wanted={}
                for line in text.splitlines():
                    k,_,value=line.partition(':')
                    if k in ('VmRSS','RssAnon','RssFile','RssShmem','VmLck'):wanted[k]=int(value.split()[0])*1024
                # Recheck lifetime so a reused PID cannot inherit old metadata.
                stat=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(stat[19])!=v.get('start_ticks'):continue
                processes.append(dict(pid=v['pid'],start_ticks=v['start_ticks'],comm=v.get('comm'),cgroup=v.get('cgroup'),**wanted))
            except (OSError,ValueError,IndexError):continue
        try:groups=memory_groups(self.cg/'system.slice')
        except OSError as e:groups=[dict(error=type(e).__name__)]
        extra=dict(time=time.time(),services=groups,processes=sorted(processes,key=lambda x:-x.get('VmRSS',0))[:24])
        with Path(self.log.name).with_name('memory_attribution.jsonl').open('a') as f:
            f.write(json.dumps(extra)+'\n')
        return result
