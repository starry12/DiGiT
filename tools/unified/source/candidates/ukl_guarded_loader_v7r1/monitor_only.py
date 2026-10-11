"""Three minutes of the actual host monitor, without CUDA, graph reads or probes."""
import fcntl
import json
from pathlib import Path
import os
import time

from . import protocol as P
from .guard import collect, Guard, AdmissionWindow, KernelCursor
from .start import append, write
from candidates.ukl_memory_ladder_v5.observer import Observer


def main():
    digest = P.verify_manifest()
    P.binding()  # Metadata only; no graph payload opened.
    _, source_device = P.source_device()
    with P.LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = P.OUT / ('monitor_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
        out.mkdir()
        started = time.monotonic()
        deadline = started + 180
        samples = 0
        first_error = None
        alerts = 0
        quiet_observed = False
        observations = {}
        extrema = {}
        kernel = KernelCursor()
        guard = Guard(require_cgroup=False)
        admission = AdmissionWindow(P.budget()['host_min'], source_devices=(source_device,))
        print('Monitor-only qualification: '+str(out), flush=True)
        with (out/'monitor.jsonl').open('x') as log, (out/'kernel.jsonl').open('x') as klog, (out/'attribution.jsonl').open('x') as alog:
            observer = Observer(alog)
            last_attribution = last_notice = started
            baseline = kernel.establish()
            append(klog, dict(time=time.time(), **baseline))
            if not baseline['ok']:
                first_error = 'kernel_baseline: '+str(baseline.get('error'))
            while True:
                sample = collect()
                if time.monotonic()-last_attribution >= 5:
                    observer.sample(sample.get('memory', {}))
                    last_attribution = time.monotonic()
                decision = guard.check(sample)
                quiet = admission.check(sample)
                evidence = kernel.poll()
                samples += 1
                alerts += len(evidence.get('alerts', []))
                quiet_observed = quiet_observed or quiet.get('ready', False)
                append(log, dict(sample=sample, decision=decision, admission=quiet))
                append(klog, dict(time=time.time(), **evidence))
                for observation in decision.get('observations', []):
                    label = observation if isinstance(observation, str) else json.dumps(observation, sort_keys=True)
                    observations[label] = observations.get(label, 0)+1
                for key, value in sample.get('memory', {}).items():
                    old = extrema.setdefault(key, dict(min=value, max=value))
                    old['min'], old['max'] = min(old['min'], value), max(old['max'], value)
                if first_error is None:
                    if not decision['ok']:
                        first_error = 'guard: '+str(decision.get('reasons'))
                    elif not quiet['ok']:
                        first_error = 'admission_monitor: '+str(quiet.get('reasons'))
                    elif not evidence['ok']:
                        first_error = 'kernel: '+str(evidence.get('error'))
                now = time.monotonic()
                if now-last_notice >= 30:
                    print(json.dumps(dict(elapsed_seconds=round(now-started, 1), samples=samples,
                        first_error=first_error, quiet_window_seen=quiet_observed)), flush=True)
                    last_notice = now
                if now >= deadline:
                    break
                time.sleep(min(1, deadline-now))
        elapsed = time.monotonic()-started
        if P.verify_manifest() != digest:
            first_error = first_error or 'Source manifest changed during qualification'
        result = dict(passed=first_error is None and samples >= 120 and elapsed >= 180,
            scope='monitor_only_no_graph_no_cuda', boot_id=kernel.boot_id,
            manifest_sha256=digest, duration_seconds=elapsed, samples=samples,
            gpu_called=False, graph_loaded=False, raw_ssd_access=False,
            quiet_window_seen=quiet_observed, kernel_alert_count=alerts,
            first_error=first_error, observations=observations, memory_bytes=extrema,
            output=str(out), full_graph_enabled=False)
        write(out/'acceptance.json', result)
        write(P.OUT/'monitor_check.json', result)
        print(json.dumps(result, indent=2), flush=True)
        if not result['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
