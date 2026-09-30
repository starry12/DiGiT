import argparse,os
from pathlib import Path
from .common import *
def main():
    a=argparse.ArgumentParser();a.add_argument('--variant',choices=['digit_default','digit_cpu2'],required=True);a.add_argument('--probe',choices=['off','on'],required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root GPU2 worker required');check_ready()
    from ae.common import check_device
    from candidates.uks_revpr_diagnostic_v1 import worker as base
    from .instrument import Probe
    base.admission();check_device();args.output.mkdir(exist_ok=True,parents=True)
    original=base.graph_sampler;probes=[]
    def graph_sampler(*a,**k):
        graph,arrays,artifact,sampler=original(*a,**k);probes.append(Probe(sampler,args.probe=='on'));return graph,arrays,artifact,sampler
    base.graph_sampler=graph_sampler
    try:report=base.run(args.variant,args.output,'stages')
    finally:base.graph_sampler=original
    require(len(probes)==1,'Unexpected sampler count');detail=probes[0].finish();check_ready()
    write(args.output/'report.json',dict(report,source_sha256=verify(),sampling_detail=detail,probe=args.probe,raw_ssd_writes=False,performance_result=False))
if __name__=='__main__':main()
