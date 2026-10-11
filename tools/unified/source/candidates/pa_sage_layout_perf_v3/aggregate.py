"""Aggregate only 15 completed, same-protocol point runs; retain slow results."""
import argparse,math
from .common import Path,read,sha,write,require,HERE

def collect(index,root):
    require(index['schema']=='digit-layout-performance-grid-v2' and len(index['points'])==15,'Need the full 15-point index')
    result=[];reference=None;seen=set()
    for item in index['points']:
        key=item['point']['id'];require(key not in seen,'Duplicate point');seen.add(key)
        require(sha(item['protocol'])==item['protocol_sha256'],'Point configuration changed')
        directory=Path(root)/key;s=read(directory/'status.json');summary=read(directory/'full_summary.json')
        require(s['passed'] and s['complete'] and s['stage']=='complete' and s['point']==item['point'],'Point not complete: '+key)
        require(s['candidate_sha256']==sha(HERE/'manifest.json'),'Wrong execution candidate')
        require([(w['mode'],w['arm'],w['returncode']) for w in s['workers']]==[('smoke','digit_full',0),('full','digit_full',0)],'Missing successful workers')
        report=directory/'full_digit_full_accepted.json';r=read(report)
        require(summary['passed'] and summary['smoke'] is False and summary['point']==r['point']==item['point'],'Wrong summary point/mode')
        require(summary['report_sha256']['digit_full']==sha(report),'Accepted report changed')
        require(summary['protocol_sha256']==r['protocol_sha256']==item['protocol_sha256'],'Wrong point protocol')
        require(r['passed'] and not r['source_only'] and not r['smoke'] and len(r['epochs'])==1 and r['test'] is None,'Incomplete native report')
        from .validation import no_evaluation_check
        no_evaluation_check(r,directory/'full/digit_full')
        require(r['candidate_sha256']==summary['candidate_sha256']==s['candidate_sha256'],'Report candidate differs')
        pair=(r['initial_parameters_sha256'],r['initial_dgl_rng'],[e['root_sha256'] for e in r['epochs']])
        if reference is None:reference=pair
        else:require(pair==reference,'Different initialization, training order')
        seconds=sum(e['train_seconds'] for e in r['epochs'])
        require(math.isfinite(seconds) and seconds>0,'Invalid training duration')
        result.append(dict(point=item['point'],mean_training_epoch_seconds=seconds,
            order_excluded_mean_seconds=sum(e['train_seconds']-e['order_seconds'] for e in r['epochs']),
            validation_seconds=0,test_seconds=None,test_accuracy=None,
            io=r['io_accounting_training'],accepted_report_sha256=sha(report)))
    baseline=next(v['mean_training_epoch_seconds'] for v in result if v['point']['id']=='g2_r20')
    for value in result:value['speedup_vs_fresh_g2_r20']=baseline/value['mean_training_epoch_seconds']
    return dict(passed=True,points=result,scope='15 fresh seed0 runs, one complete first epoch per point; one repeat; no accuracy or steady-state claim')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Preserve earlier aggregation');write(a.output,collect(read(a.index),a.runs))
if __name__=='__main__':main()
