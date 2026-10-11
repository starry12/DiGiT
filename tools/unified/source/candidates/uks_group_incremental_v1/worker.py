import argparse,os
from pathlib import Path
from .common import *
def main():
    a=argparse.ArgumentParser();a.add_argument('--kernel',choices=['legacy','incremental'],required=True);a.add_argument('--probe',choices=['off','on'],required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root GPU2 worker required');check_ready()
    os.environ['UKS_GROUP_KERNEL']=args.kernel
    from ae.common import check_device
    from candidates.uks_revpr_diagnostic_v1 import worker as base
    from .instrument import Probe
    base.admission();check_device();args.output.mkdir(exist_ok=True,parents=True)
    original=base.graph_sampler;probes=[]
    def graph_sampler(*a,**k):
        graph,arrays,artifact,sampler=original(*a,**k)
        import digit.sampler as module
        module._cuda_extension=extension()
        probes.append(Probe(sampler,args.probe=='on'));return graph,arrays,artifact,sampler
    base.graph_sampler=graph_sampler
    try:report=base.run('digit_cpu2',args.output,'stages' if args.probe=='on' else 'off')
    finally:base.graph_sampler=original
    require(len(probes)==1,'Unexpected sampler count');detail=probes[0].finish();check_ready()
    write(args.output/'report.json',dict(report,source_sha256=verify(),sampling_detail=detail,kernel=args.kernel,probe=args.probe,raw_ssd_writes=False,performance_result=args.probe=='off'))
if __name__=='__main__':main()
