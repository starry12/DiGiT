"""Verify controller-owned monitoring independently of worker checkpoints."""
import json
from pathlib import Path

def monitor_evidence(output,status,arm,resources):
    output=Path(output)
    summary=json.loads((output/'external_gpu/summary.json').read_text())
    samples=[json.loads(line) for line in (output/'external_gpu/samples.jsonl').read_text().splitlines()]
    worker=next(w for w in status['workers'] if w['arm']==arm)
    if not (summary['passed'] and summary['complete'] and status['external_monitor_returncode']==0):
        raise RuntimeError('External monitor incomplete/failed')
    if summary['pid']!=status['external_monitor']['pid'] or summary['parent_pid']!=status['pid']:
        raise RuntimeError('External monitor is not owned by the controller')
    if summary['pid'] in [w['pid'] for w in status['workers']]:raise RuntimeError('Monitor and training are not isolated')
    if summary['peak_rss_bytes']>64*2**20:raise RuntimeError('Monitor exceeded its 64 MiB host limit')
    if resources['mode']!='checkpoints_only_external_sampler' or resources['background_monitor_in_worker'] or resources['monitor_samples']!=0:
        raise RuntimeError('Worker still contains a continuous resource monitor')
    if resources['worker_pid']!=worker['pid']:raise RuntimeError('Wrong worker checkpoint identity')
    points=[r for r in samples if 'error' not in r and worker['started_unix']<=r['time_unix']<=worker['finished_unix']]
    if not points:raise RuntimeError('No independent GPU samples during worker execution')
    if any(r['monitor_pid']!=summary['pid'] or r['monitor_parent_pid']!=status['pid'] for r in points):
        raise RuntimeError('Wrong GPU sample provenance')
    return dict(monitor_pid=summary['pid'],worker_pid=worker['pid'],sample_count=len(points),
        monitor_peak_rss_bytes=summary['peak_rss_bytes'],
        observed_peak_device_used_bytes=max([resources['observed_peak_device_used_bytes']]+[r['device_used_bytes'] for r in points]))
