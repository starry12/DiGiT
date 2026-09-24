"""One-time reuse of individually accepted smoke evidence, never full epochs."""
import hashlib, importlib, json
from pathlib import Path

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise RuntimeError(message)

def verify(control, bindings, package):
    control = Path(control)
    if not (control / 'resume_smoke.json').exists() or (control / 'state/resume_consumed_setup_v3.json').exists(): return {}, None
    plan = read(control / 'resume_smoke.json')
    source = Path(plan['source'])
    require(source.parent == Path('/srv/digit-ae/ablation-results'), 'Resume source outside AE output root')
    require(plan['arms'] == ['gids', 'gr'], 'Only accepted GIDS/GR smoke can be reused')
    require(sha(control / 'snapshot_manifest.json') == plan['snapshot_sha256'], 'Resume snapshot changed')
    for name, digest in plan['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe resume evidence path')
        require(sha(source / relative) == digest, 'Resume evidence changed: ' + name)
    state = read(source / 'status.json')
    require(state['snapshot_sha256'] == plan['snapshot_sha256'] and state['gpu'] == 2 and not state['selftest'], 'Source service identity differs')
    from ae.pa_sage.monitor_validation import monitor_evidence
    reports = {}
    for arm in plan['arms']:
        key = 'gr' if arm == 'gr' else 'gids'
        binding_path = source / ('inputs_' + key + '.json')
        require(read(binding_path) == bindings[key], 'Resume input binding differs')
        folder = source / 'smoke' / arm
        accepted = read(source / ('smoke_' + arm + '_accepted.json'))
        raw = read(folder / 'report.json')
        require({k:v for k,v in accepted.items() if k != 'external_monitor'} == raw, 'Accepted/raw report differs')
        receipt = read(folder / 'worker_ready.json')
        require(receipt['passed'] and receipt['report_sha256'] == sha(folder / 'report.json'), 'Source worker receipt differs')
        require(raw['input_binding_sha256'] == sha(binding_path), 'Source input receipt differs')
        workers = [w for w in state['workers'] if w['arm'] == arm and w['mode'] == 'smoke']
        require(len(workers) == 1 and workers[0]['status'] == 'complete' and workers[0]['returncode'] == 0, 'Source worker not accepted')
        mon = source / ('smoke_monitor_' + arm)
        ms = read(mon / 'external_gpu/summary.json')
        release = read(folder / 'release_worker.json')
        require(ms['errors'] == [] and release['monitor_stopped'] and release['monitor_returncode'] == 0, 'Source monitor failed')
        ownership = dict(pid=state['pid'], workers=workers,
            external_monitor=read(mon / 'external_gpu/ready.json'), external_monitor_returncode=release['monitor_returncode'])
        evidence = monitor_evidence(mon, ownership, arm, read(folder / 'resources.json'))
        require(evidence == accepted['external_monitor'], 'Source monitoring evidence differs')
        importlib.import_module(package(arm) + '.validation').report_check(accepted, arm, True, bindings[key], folder)
        reports[arm] = accepted
    return reports, dict(source=str(source), arms=plan['arms'], plan_sha256=sha(control / 'resume_smoke.json'),
        scope='Individually accepted smoke only; no full epoch reused', source_snapshot_sha256=plan['snapshot_sha256'])
