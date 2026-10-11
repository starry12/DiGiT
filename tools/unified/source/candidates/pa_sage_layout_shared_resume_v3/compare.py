"""Predeclared calibration gate; never select the faster repeat for the grid."""
import statistics
from .common import read,sha,require,Path,write

TIME_TOLERANCE=.10
PHYSICAL_IO_TOLERANCE=.05

def relative(a,b):
    require(a>0 and b>0,'Nonpositive comparison metric');return abs(a/b-1.)

def collect(root):
    root=Path(root)
    paths=[('real',root/'calibration/real_first'),('shared',root/'native/g2_r20'),
           ('shared',root/'calibration/shared_repeat'),('real',root/'calibration/real_second')]
    reports=[];smokes=[];hashes={}
    for mode,path in paths:
        s=read(path/'status.json');summary=read(path/'full_summary.json');r=read(path/'full_digit_full_accepted.json')
        require(s['complete'] and s['passed'] and summary['passed'] and r['passed'],'Unaccepted pilot worker')
        require(r['feature_mode']==mode and not r['smoke'] and len(r['epochs'])==1 and r['test'] is None,'Wrong pilot mode/extent')
        require(summary['report_sha256']['digit_full']==sha(path/'full_digit_full_accepted.json'),'Report changed')
        require([(w['mode'],w['returncode']) for w in s['workers']]==[('smoke',0),('full',0)],'Bad pilot exits')
        from .validation import no_evaluation_check
        no_evaluation_check(r,path/'full/digit_full')
        smoke_summary=read(path/'smoke_summary.json')
        require(smoke_summary['passed'] and smoke_summary['report_sha256']['digit_full']==sha(path/'smoke_digit_full_accepted.json'),'Smoke report changed')
        reports.append(r);smokes.append(read(path/'smoke_digit_full_accepted.json'))
        hashes[str(path)]=sha(path/'full_digit_full_accepted.json')
    first=reports[0];exact=['initial_parameters_sha256','initial_dgl_rng','point','layout_manifest_sha256','overlay_receipt_sha256','policy','candidate_sha256']
    for r in reports:
        for key in exact:require(r[key]==first[key],'Unpaired pilot field: '+key)
        a=r['epochs'][0];b=first['epochs'][0]
        for key in ('updates','root_sha256','shapes'):require(a[key]==b[key],'Different training sampling: '+key)
        require(len(a['windows'])==len(b['windows']),'Different sampling window counts')
        for x,y in zip(a['windows'],b['windows']):
            for key in ('start_update','end_update','input_nodes','target_edges','actual_edges','underfilled_owners','shortfall_edges','group_edges'):
                require(x[key]==y[key],'Sampling/window count differs: '+key)
        for key in ('cpu','gpu_ssd'):require(a['training']['feature'][key]==b['training']['feature'][key],'Feature request count differs')
    for s in smokes[1:]:
        require(len(s['audits'])==len(smokes[0]['audits'])==4,'Missing smoke audits')
        for a,b in zip(s['audits'],smokes[0]['audits']):
            for key in ('inputs','outputs','blocks','storage_rows','storage_flags','cuda_rng','dgl_rng'):
                require(a[key]==b[key],'Different smoke sampling/request trace: '+key)
    times={m:[sum(e['train_seconds']-e['order_seconds'] for e in r['epochs']) for r in reports if r['feature_mode']==m] for m in ('real','shared')}
    medians={m:statistics.median(v) for m,v in times.items()}
    timing_delta=relative(medians['shared'],medians['real'])
    spreads={m:relative(max(v),min(v)) for m,v in times.items()}
    io_deltas=[]
    for r in reports:
        for key in ('ssd_primary_bytes','ssd_completed_bytes','ssd_useful_bytes'):
            io_deltas.append(relative(r['io_accounting_training'][key],first['io_accounting_training'][key]))
    passed=timing_delta<=TIME_TOLERANCE and max(spreads.values())<=TIME_TOLERANCE and max(io_deltas)<=PHYSICAL_IO_TOLERANCE
    result=dict(passed=passed,paired_sampling_passed=True,order=['real','shared','shared','real'],epochs_per_worker=1,
        timings_order_excluded=times,medians_order_excluded=medians,relative_median_difference=timing_delta,
        within_mode_spreads=spreads,max_physical_io_relative_difference=max(io_deltas),time_tolerance=TIME_TOLERANCE,
        physical_io_tolerance=PHYSICAL_IO_TOLERANCE,accepted_reports=hashes,grid_baseline='first shared run, fixed before measurement',
        scope='Engineering calibration at g2/r20; both feature contents and SSD base extent change. Not proof of universal equivalence or accuracy.')
    write(root/'calibration_review.json',result)
    require(passed,'Calibration differs beyond declared limits; retain results and stop before the grid')
    return result
