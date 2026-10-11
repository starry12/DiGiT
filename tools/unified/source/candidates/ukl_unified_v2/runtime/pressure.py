import math
import os
import time
from pathlib import Path

LIMITS={'some':10.0,'full':5.0}


def rates(before,after,seconds):
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('Invalid pressure interval')
    result={}
    for kind in LIMITS:
        a,b=before[kind]['total'],after[kind]['total']
        if not all(math.isfinite(x) and x>=0 for x in (a,b)) or b<a:
            raise ValueError('Pressure counter reset or invalid')
        result[kind]=(b-a)/1e6/seconds*100
    return result


def high(value):
    return any(value[k]>=limit for k,limit in LIMITS.items())


def observe():
    from .guard_base import _psi
    from .phase import CGROOT,membership
    leaf=CGROOT/membership(os.getpid()).lstrip('/')
    return dict(monotonic=time.monotonic(),host=_psi(Path('/proc/pressure/memory').read_text()),
                worker=_psi((leaf.parent/'memory.pressure').read_text()))


def interval(before,after):
    seconds=after['monotonic']-before['monotonic']
    values={name:rates(before[name],after[name],seconds) for name in ('host','worker')}
    return dict(seconds=seconds,stall_percent=values,passed=not any(high(v) for v in values.values()))


def valid_receipt(value,warmup):
    try:
        wanted={'training','warmup'} if warmup else {'training'}
        if value['passed'] is not True or set(value['gates'])!=wanted:return False
        if not all(g['passed'] is True and g['quiet_seconds']>=5 and g['wait_seconds']<=300 for g in value['gates'].values()):return False
        measured=[v for v in value['windows'] if v['phase']=='training']
        if not measured:return False
        for v in measured:
            if v['passed'] is not True or not math.isfinite(v['seconds']) or v['seconds']<=0:return False
            if set(v['stall_percent'])!={'host','worker'}:return False
            for values in v['stall_percent'].values():
                if any(not math.isfinite(values[k]) or values[k]<0 for k in LIMITS) or high(values):return False
        return True
    except (KeyError,TypeError,ValueError):return False


def settle(check,event,read=observe,sleep=time.sleep,timeout=300,quiet_seconds=5):
    before=read();start=before['monotonic'];quiet=None
    while True:
        check();sleep(1);current=read();value=interval(before,current)
        average_high=any(high({k:current[name][k]['avg10'] for k in LIMITS}) for name in ('host','worker'))
        healthy=value['passed'] and not average_high
        quiet=(current['monotonic'] if quiet is None else quiet) if healthy else None
        elapsed=current['monotonic']-start
        event('memory_pressure_settling',wait_seconds=elapsed,recent=value,avg10_high=average_high)
        if quiet is not None and current['monotonic']-quiet>=quiet_seconds:
            event('memory_pressure_ready',wait_seconds=elapsed)
            return dict(passed=True,wait_seconds=elapsed,quiet_seconds=quiet_seconds)
        if elapsed>=timeout:raise RuntimeError('Memory pressure did not recover before training')
        before=current
