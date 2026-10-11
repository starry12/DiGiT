"""Unprivileged one-minute queue monitor; never starts training without sudo."""
import fcntl
import subprocess
from .common import *
from .controller import active_ae


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            status=dict(pid=os.getpid(),updated_unix=time.time(),poll_seconds=60)
            try:
                latest=OUT/'latest.json'
                if latest.exists():
                    value=read(latest)
                    status.update(stage='handed_to_controller',controller=value)
                    write(OUT/'watch_status.json',status)
                    print(json.dumps(status),flush=True)
                    return
                active=active_ae()
                if active:
                    status.update(stage='waiting_for_reviewers',blockers=active,
                                  training_controller_started=False,requires_sudo_launch=True)
                else:
                    result=subprocess.run(['sudo','-n',PY,'-B','-m',MODULE+'.start','--execute'],
                        cwd=ROOT,capture_output=True,text=True,timeout=45)
                    status.update(stage='controller_launched' if result.returncode==0 else 'waiting_for_sudo_launch',
                                  launch_returncode=result.returncode,
                                  launch_output=(result.stdout+result.stderr)[-6000:])
            except Exception as error:
                status.update(stage='monitor_error',error=str(error))
            write(OUT/'watch_status.json',status)
            print(json.dumps(status),flush=True)
            time.sleep(60)


if __name__=='__main__':main()
