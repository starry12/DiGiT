"""Reconcile actual 512B/1KiB commands instead of the old 4KiB slot counters."""
from .common import require
from candidates.io_accounting_v1.accounting import decode,useful_interval,validate_region
from candidates.pa_sage_mixed_group_v1.common import check_pair_counts

def install_runner(r):
    import torch
    from baseline_metrics import gpu_cache_interval,device_io_interval
    def snapshot(loader):
        torch.cuda.synchronize()
        return dict(gpu=loader.get_gpu_cache_stats(),device=loader.get_device_io_stats(),
            feature=loader.get_feature_access_stats(),useful_io=decode(list(loader.BAM_FS.get_useful_io_stats())),
            pair_counts=list(loader.BAM_FS.get_pair_counts()),paired=loader.pair_enabled,
            sampling_seconds=loader.sample_time,feature_seconds=loader.feature_time)
    def interval(a,b,rows):
        require(a['paired']==b['paired'],'Read mode changed')
        gpu=gpu_cache_interval(a['gpu'],b['gpu'])
        device=device_io_interval(a['device'],b['device'],maxima_scope='epoch_cumulative')
        feature={k:b['feature'][k]-a['feature'][k] for k in a['feature']}
        require(sum(feature.values())==rows and min(feature.values())>=0 and gpu['requests']==feature['gpu_ssd'],'Feature route mismatch')
        counts=[y-x for x,y in zip(a['pair_counts'],b['pair_counts'])]
        check_pair_counts(counts,device,b['paired'],require_pair=False)
        value=dict(gpu=gpu,device=device,feature=feature,pair_counts=counts,
            mixed=dict(enabled=b['paired'],reconciled=True,request_bytes=[512,1024] if b['paired'] else [512]),
            useful_io=useful_interval(a['useful_io'],b['useful_io'],feature),feature_row_bytes=512,
            sampling_seconds=b['sampling_seconds']-a['sampling_seconds'],feature_seconds=b['feature_seconds']-a['feature_seconds'],
            reconciliation=dict(reconciled=True,device_counters_checked=True,kind='actual_pair_command_histogram'))
        require(device['primary_bytes']==gpu['inserts']*512,'GPU insert bytes differ from SSD primary bytes')
        return value
    r.snapshot=snapshot;r.interval=interval
