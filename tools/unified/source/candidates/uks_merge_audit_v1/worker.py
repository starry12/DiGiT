import argparse,os
from pathlib import Path
from .common import *
def main():
    a=argparse.ArgumentParser();a.add_argument('--job',choices=['merge_audit','digit_profile','gids_profile'],required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root GPU2 required');check_ready()
    from candidates.uks_mixed_1k2k_v1 import worker as base
    from ae.common import check_device
    base.admission();check_device();args.output.mkdir(exist_ok=True,parents=True)
    old=base.graph_sampler;audits=[];probes=[]
    def build(*a,**k):
        result=old(*a,**k);sampler=result[3]
        if args.job=='merge_audit':
            from .audit import Audit
            audits.append(Audit(sampler))
        elif args.job=='digit_profile':
            from candidates.uks_group_incremental_v1.instrument import Probe
            probes.append(Probe(sampler,True))
        return result
    base.graph_sampler=build
    try:r=base.run('gids_default' if args.job=='gids_profile' else 'digit_cpu2',args.output,'off' if args.job=='merge_audit' else 'stages')
    finally:base.graph_sampler=old
    detail=audits[0].finish() if audits else probes[0].finish() if probes else {}
    if audits:require(detail['totals']['mapped_pairs_both_cpu_cold']==r['training']['request_sizes']['pair_requests'],'CPU eligibility/native pair count mismatch')
    check_ready();write(args.output/'report.json',dict(r,source_sha256=verify(),diagnostic=detail,job=args.job,performance_result=False))
if __name__=='__main__':main()
