import contextlib,fcntl,os,subprocess,time
from .common import require

LOCKS = ('/run/digit-ae-selfservice/exclusive.lock',
         '/tmp/digit-pa-bidir-controller.lock', '/tmp/digit-pa-sage-512b-pair-controller.lock', '/tmp/digit-pa-sage-libnvm0.lock')


def active_ae():
    result = subprocess.run(['systemctl', 'list-units', 'digit-ae*', '--all',
                             '--no-pager', '--plain', '--no-legend'],
                            capture_output=True, text=True, timeout=15, check=True)
    active = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[2] in ('active', 'activating', 'deactivating') and fields[3] != 'exited':
            active.append(fields[0])
    return active + reviewer_chains()


def reviewer_chains():
    # Wait for the reviewer's whole queue, including sleeps between UKL and CL.
    result=subprocess.run(['ps','-u','atc27_ae','-o','pid=,args='],
                          capture_output=True,text=True,timeout=15)
    require(result.returncode in (0,1),'Cannot inspect reviewer queue')
    return parse_reviewer_chains(result.stdout)


def parse_reviewer_chains(text):
    import re
    active=[]
    for line in text.splitlines():
        parts=line.split()
        if len(parts)>1 and any(re.fullmatch(r'/home/atc27_ae/ae_chain[0-9]*\.sh',arg) for arg in parts[1:]):
            active.append('reviewer_queue_pid_'+parts[0])
    return active


@contextlib.contextmanager
def exclusion(save):
    while True:
        stack = contextlib.ExitStack()
        try:
            active = active_ae()
            if active:
                raise BlockingIOError('Active reviewer units: ' + ', '.join(active))
            handles = []
            for path in LOCKS:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                stack.callback(os.close, fd)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handles.append(fd)
            active = active_ae()
            if active:
                raise BlockingIOError('Reviewer started while admitting: ' + ', '.join(active))
        except BlockingIOError as error:
            stack.close()
            save(stage='waiting_for_reviewers',reason=str(error))
            time.sleep(60)
            continue
        except BaseException:
            stack.close()
            raise
        try:
            yield handles[-1]
        finally:
            stack.close()
        return


