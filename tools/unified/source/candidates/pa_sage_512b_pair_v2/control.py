"""Two native smokes, followed by one fresh full epoch for each system."""
import argparse
import csv
import fcntl
import subprocess
import traceback
from .common import *


def gpu_admission(folder, stage):
    query = subprocess.run(['nvidia-smi', '-i', '2',
        '--query-gpu=uuid,name,memory.total,memory.used', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, timeout=20, check=True)
    fields = [x.strip() for x in query.stdout.strip().split(',')]
    require(len(fields) == 4 and fields[0] == GPU_UUID, 'Physical GPU identity mismatch')
    require(int(fields[3]) < 1024, 'GPU 2 occupied; refusing to interfere')
    write(folder / (stage + '_gpu_admission.json'), dict(query=fields, observed_unix=time.time(),
                                                       continuous_monitoring=False))


def stages(protocol):
    return [(kind+'_'+arm,kind=='smoke',arm) for kind in ('smoke','full') for arm in protocol['arms']]


def run(folder):
    source = verify()
    require(os.geteuid() == 0, 'Native controller requires local administrator execution')
    require(not folder.exists(), 'Preserve previous run output')
    folder.mkdir(parents=True)
    state = dict(passed=False, complete=False, stage='admission', directory=str(folder),
                 candidate_sha256=source, started_unix=time.time(), completed_stages=[])

    def status(stage, **extra):
        state.update(stage=stage, updated_unix=time.time(), **extra)
        write(OUT / 'status.json', state); write(folder / 'status.json', state)
        print(json.dumps(state), flush=True)

    try:
        check_inputs(read(OUT / 'inputs.json'))
        p = read(HERE / 'protocol.json')
        full_reports = []; rows = []; evidence = []
        for stage, smoke, arm in stages(p):
            status(stage, arm=arm)
            gpu_admission(folder, stage)
            dest = folder / stage
            cmd = [PY, '-B', '-u', '-m', MODULE + '.worker', '--output', str(dest), '--arm', arm]
            if smoke:
                cmd.append('--smoke')
            write(folder / (stage + '_command.json'), cmd)
            with (folder / (stage + '.log')).open('x') as log:
                worker = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write(folder / (stage + '_exit.json'), dict(returncode=worker.returncode, finished_unix=time.time()))
            require(worker.returncode == 0, stage + ' failed; see ' + str(folder / (stage + '.log')))
            receipt = read(dest / 'accepted.json'); report = read(dest / 'report.json')
            require(receipt['passed'] and receipt['report_sha256'] == sha(dest / 'report.json') and
                    receipt['candidate_sha256'] == source and report['candidate_sha256'] == source and
                    receipt['binary_sha256'] == sha(BINARY), 'Worker acceptance identity mismatch')
            validate_completion(report)
            require(report['smoke'] == smoke and report['arm'] == arm, 'Worker mode mismatch')
            evidence.append(dict(stage=stage, report=str(dest/'report.json'), report_sha256=sha(dest/'report.json'),
                                 receipt_sha256=sha(dest/'accepted.json'), returncode=worker.returncode))
            state['completed_stages'].append(stage)
            if not smoke:
                full_reports.append(report)
                rows.append(dict(stage=stage, arm=arm, examples=report['examples'], updates=report['updates'],
                    **{k:v for k,v in report['timing'].items() if k.endswith('_seconds')},
                    logical_feature_requests=report['logical_feature_requests'],
                    gpu_cache_hit_rate=report['gpu_cache']['hit_rate'],
                    ssd_read_bytes=report['device_io']['completed_bytes']))
            status(stage + '_accepted')
        require(verify() == source, 'Candidate changed during experiment')
        check_inputs(read(OUT / 'inputs.json'))
        summary = compare(*full_reports)
        summary.update(run_directory=str(folder), evidence=evidence, rows=rows)
        write(folder / 'summary.json', summary); write(OUT / 'summary.json', summary)
        for dest in (folder / 'summary.csv', OUT / 'summary.csv'):
            with dest.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        status('complete', passed=True, complete=True, finished_unix=time.time(),
               e2e_speedup=summary['speedup'])
    except BaseException as error:
        status('failed', error=str(error), finished_unix=time.time(), complete=False, passed=False)
        write(folder / 'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with open('/tmp/digit-pa-sage-512b-pair-controller.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.output)


if __name__ == '__main__':
    main()
