"""Run tail-sensitive native acceptance then a fresh complete training epoch."""
import argparse
import subprocess
import traceback
from .common import *


def run(folder):
    source = verify()
    require(os.geteuid() == 0, 'Native controller requires local administrator execution')
    require(not folder.exists(), 'Preserve previous run output')
    folder.mkdir(parents=True)
    state = dict(passed=False, complete=False, stage='admission', directory=str(folder),
                 candidate_sha256=source, started_unix=time.time())

    def status(stage, **extra):
        state.update(stage=stage, updated_unix=time.time(), **extra)
        write(OUT / 'status.json', state)
        write(folder / 'status.json', state)
        print(json.dumps(state), flush=True)

    try:
        check_inputs(read(OUT / 'inputs.json'))
        # One startup query, no polling during preparation or training.
        query = subprocess.run(['nvidia-smi', '-i', '2',
            '--query-gpu=uuid,name,memory.total,memory.used', '--format=csv,noheader,nounits'],
            text=True, capture_output=True, timeout=20, check=True)
        fields = [x.strip() for x in query.stdout.strip().split(',')]
        require(len(fields) == 4 and fields[0] == GPU_UUID, 'Physical GPU identity mismatch')
        require(int(fields[3]) < 1024, 'GPU 2 is occupied; do not interfere with another task')
        write(folder / 'startup_gpu.json', dict(query=fields, continuous_monitoring=False))
        for stage, smoke in [('smoke', True), ('full', False)]:
            status(stage)
            dest = folder / stage
            cmd = [PY, '-B', '-u', '-m', MODULE + '.worker', '--output', str(dest)]
            if smoke:
                cmd.append('--smoke')
            write(folder / (stage + '_command.json'), cmd)
            with (folder / (stage + '.log')).open('x') as log:
                worker = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write(folder / (stage + '_exit.json'), dict(returncode=worker.returncode, finished_unix=time.time()))
            require(worker.returncode == 0, stage + ' worker failed; see ' + str(folder / (stage + '.log')))
            receipt = read(dest / 'accepted.json')
            require(receipt['passed'] and receipt['report_sha256'] == sha(dest / 'report.json') and
                    receipt['candidate_sha256'] == source, 'Worker acceptance receipt mismatch')
            report = read(dest / 'report.json')
            validate_completion(report)
            state[stage + '_report_sha256'] = sha(dest / 'report.json')
        require(verify() == source, 'Candidate changed during experiment')
        full = read(folder / 'full/report.json')
        summary = dict(passed=True, complete=True, updates=full['updates'], examples=full['examples'],
            e2e_seconds=full['timing']['e2e_seconds'],
            legacy_overlapping_sum_seconds=full['timing']['legacy_overlapping_sum_seconds'],
            feature_read_seconds=full['timing']['native_merged_read_seconds'],
            logical_feature_requests=full['logical_feature_requests'],
            gpu_cache_hit_rate=full['gpu_cache']['hit_rate'], device_io=full['device_io'],
            graph='directed normalized', cache_page_bytes=512, window_buffer=True, accumulator=True,
            candidate_sha256=source, run_directory=str(folder), report_sha256=sha(folder/'full/report.json'))
        write(folder / 'summary.json', summary)
        write(OUT / 'summary.json', summary)
        status('complete', complete=True, passed=True, finished_unix=time.time(),
               e2e_seconds=summary['e2e_seconds'], updates=full['updates'])
    except BaseException as error:
        status('failed', error=str(error), finished_unix=time.time(), complete=False, passed=False)
        write(folder / 'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
