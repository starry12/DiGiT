"""Aggregate only 15 completed, same-protocol point runs; retain slow results."""
import argparse,math
from .common import Path,read,sha,write,require,HERE,verification_policy

def collect(index,root):
    require(index['schema']=='digit-layout-performance-grid-v2' and len(index['points'])==15,'Need the full 15-point index')
    result=[];reference=None;seen=set()
    for item in index['points']:
        key=item['point']['id'];require(key not in seen,'Duplicate point');seen.add(key)
        require(sha(item['protocol'])==item['protocol_sha256'],'Point configuration changed')
        directory=Path(root)/key;s=read(directory/'status.json');summary=read(directory/'full_summary.json')
        require(s['passed'] and s['complete'] and s['stage']=='complete' and s['point']==item['point'],'Point not complete: '+key)
        require(s['candidate_sha256']==sha(HERE/'manifest.json'),'Wrong execution candidate')
        policy=verification_policy(item['point'])
        expected=[('smoke','digit_full',0),('full','digit_full',0)] if policy['independent_native_smoke'] else [('full','digit_full',0)]
        require([(w['mode'],w['arm'],w['returncode']) for w in s['workers']]==expected,'Missing successful workers')
        report=directory/'full_digit_full_accepted.json';r=read(report)
        require(summary['passed'] and summary['smoke'] is False and summary['point']==r['point']==item['point'],'Wrong summary point/mode')
        require(summary['report_sha256']['digit_full']==sha(report),'Accepted report changed')
        require(summary['protocol_sha256']==r['protocol_sha256']==item['protocol_sha256'],'Wrong point protocol')
        require(r['passed'] and not r['source_only'] and not r['smoke'] and len(r['epochs'])==1 and r['test'] is None,'Incomplete native report')
        require(r['verification_policy']==summary['verification_policy']==policy,'Wrong verification scope')
        require(r['feature_mode']==summary['feature_mode']=='shared','Grid must use one shared-pool protocol')
        from .validation import no_evaluation_check
        no_evaluation_check(r,directory/'full/digit_full')
        require(r['candidate_sha256']==summary['candidate_sha256']==s['candidate_sha256'],'Report candidate differs')
        pair=(r['initial_parameters_sha256'],r['initial_dgl_rng'],[e['root_sha256'] for e in r['epochs']])
        if reference is None:reference=pair
        else:require(pair==reference,'Different initialization, training order')
        seconds=sum(e['train_seconds'] for e in r['epochs'])
        require(math.isfinite(seconds) and seconds>0,'Invalid training duration')
        windows=[w for e in r['epochs'] for w in e['windows']]
        sample_counts={key:sum(w[key] for w in windows) for key in ('target_edges','actual_edges','group_edges','shortfall_edges')}
        p=read(item['protocol']);base=Path(p['base_layout']);build=read(base/'build_receipt.json');m=read(base/'final/bundle/manifest.json')
        require(build['manifest_sha256']==r['layout_manifest_sha256']==sha(base/'final/bundle/manifest.json'),'Layout changed before aggregation')
        result.append(dict(point=item['point'],verification_policy=policy,mean_training_epoch_seconds=seconds,
            order_excluded_mean_seconds=sum(e['train_seconds']-e['order_seconds'] for e in r['epochs']),
            validation_seconds=0,test_seconds=None,test_accuracy=None,
            layout_address_span_bytes=build['payload_bytes'],padding_rows=build['padding_rows'],
            achieved_replica_fraction=build['achieved_replica_fraction'],groups=m['grouping']['num_groups'],
            sampling=sample_counts,group_sample_fraction=sample_counts['group_edges']/max(1,sample_counts['actual_edges']),
            preparation=build['preparation'],preparation_reused=item['point']['id']=='g2_r20',
            preparation_feature_payload_materialized=not build.get('metadata_only',False),
            io=r['io_accounting_training'],accepted_report_sha256=sha(report)))
    baseline=next(v['mean_training_epoch_seconds'] for v in result if v['point']['id']=='g2_r20')
    order_excluded_baseline=next(v['order_excluded_mean_seconds'] for v in result if v['point']['id']=='g2_r20')
    for value in result:
        value['speedup_vs_fresh_g2_r20']=baseline/value['mean_training_epoch_seconds']
        value['order_excluded_speedup_vs_g2_r20']=order_excluded_baseline/value['order_excluded_mean_seconds']
    return dict(passed=True,points=result,scope='15 shared physical-row proxy feature runs, one complete first epoch per point; first shared pilot is the reference; no accuracy or steady-state claim')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Preserve earlier aggregation');write(a.output,collect(read(a.index),a.runs))
if __name__=='__main__':main()
