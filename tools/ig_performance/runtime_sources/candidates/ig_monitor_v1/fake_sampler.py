"""CPU-only test child; never selected by the production monitor CLI."""
import json
import os
import sys
import time

scenario = sys.argv[1]
for line in sys.stdin:
    request = json.loads(line)
    sequence = request['sequence']
    started = time.time()
    if scenario == 'hang' or (scenario == 'hang_after_ready' and sequence > 1):
        time.sleep(60)
    if scenario == 'disconnect' or (scenario == 'disconnect_after_ready' and sequence > 1):
        sys.exit(0)
    if scenario == 'partial':
        print('{', end='', flush=True)
        time.sleep(60)
    response = dict(ok=True, sequence=sequence, sampler_pid=os.getpid(), sampler_parent_pid=os.getppid(),
                    backend='test_fake_backend', physical_gpu_index=2, physical_gpu_uuid='GPU-fake-device',
                    device_used_bytes=123456, device_total_bytes=2**30, utilization_percent=7,
                    nvml_version='fake', driver_version='fake', query_started_unix=started, query_finished_unix=time.time())
    if scenario == 'error' and sequence > 1:
        response.update(ok=False, error_kind='nvml_query_error', error='Fake query failure')
    if scenario == 'wrong_gpu':
        response['physical_gpu_index'] = 3
    if scenario == 'wrong_pid':
        response['sampler_pid'] += 1
    if scenario == 'wrong_parent':
        response['sampler_parent_pid'] += 1
    if scenario == 'wrong_sequence':
        response['sequence'] += 1
    if scenario == 'changing_uuid' and sequence > 1:
        response['physical_gpu_uuid'] = 'GPU-different-device'
    print(json.dumps(response), flush=True)

    if scenario == 'exit_after_response':
        sys.exit(3)
