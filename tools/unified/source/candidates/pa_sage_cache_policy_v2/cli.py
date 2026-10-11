"""Offline inspection/checks only. No training or presampling command is auto-run."""
import argparse
import ast
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from .common import HERE, ROOT, OUT, GRID, read, write_new, require, protected_now, verify


def cpu_check(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            os.environ.get('OMP_NUM_THREADS') == os.environ.get('OPENBLAS_NUM_THREADS') == '1',
            'Use hidden CUDA and one CPU/BLAS thread')
    state = read(GRID / 'status.json')
    require(not state['stage'].startswith(('native_', 'budget_')),
            'Defer even CPU checks during native grid measurement')
    os.nice(19)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
    resource.setrlimit(resource.RLIMIT_AS, (2*2**30, 2*2**30))
    output.mkdir(parents=True, exist_ok=False)
    before = protected_now()
    require(before == read(OUT / 'protected_before.json'), 'Protected source changed before checks')
    started = time.time()
    for file in HERE.glob('*.py'):
        ast.parse(file.read_text(), filename=str(file))
    from .tests import run
    result = run()
    command = ['/usr/bin/g++', '-std=c++11', '-O0', '-Wall', '-Wextra', '-Werror',
               str(HERE / 'native/policy_math_test.cpp'), '-o', str(output / 'policy_math_test')]
    with (output / 'cpp.log').open('x') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=15)
        subprocess.run([str(output / 'policy_math_test')], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=5)
    after = protected_now()
    require(before == after, 'Active sources/binaries changed')
    import sys
    require('torch' not in sys.modules and 'BAM_Feature_Store' not in sys.modules,
            'CPU checks unexpectedly imported a device runtime')
    write_new(output / 'protected_after.json', after)
    write_new(output / 'summary.json', dict(passed=True, regression=result,
        cpp_shared_address_math_passed=True, cpp_command=command,
        native_compiled=False, native_execution=False, cuda_initialized=False, raw_ssd_access=False,
        gpu_runtime_imported=False, protected_sources_unchanged=True,
        wall_seconds=time.time()-started, peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        compiler_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.nice(0),
        grid_before={k:state.get(k) for k in ('stage','pid','completed')},
        grid_after={k:read(GRID / 'status.json').get(k) for k in ('stage','pid','completed')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('preview', 'cpu-check', 'verify'))
    p.add_argument('--output', type=Path, default=OUT / 'cpu_checks')
    a = p.parse_args()
    if a.action == 'cpu-check':
        cpu_check(a.output)
    elif a.action == 'verify':
        print(verify())
    else:
        from .build import plan
        print(json.dumps(dict(native_source_written=True, **plan(),
            pending=['CUDA build', 'full graph rankings and independent frequency presampling',
                     'worker/monitor integration and native short acceptance', 'four full-epoch measurements']), indent=2))


if __name__ == '__main__':
    main()
