"""Replace only fixed-GPU admission; use the accepted UKS compute functions intact."""
import argparse,json,os,sys
from pathlib import Path
C=Path(__file__).resolve().parent;sys.path.insert(0,str(C));sys.path.insert(0,'/home/embed/digit')
from review import sha
from gpu_select import admission
from candidates.uks_freq_bfs_retry_v1.common import check_ready,verify,write,require
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['gids','digit'],required=True);p.add_argument('--mode',choices=['smoke','performance'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(os.geteuid()==0 and __debug__,'Fixed root service required');check_ready();selected=admission()
    from ae.common import check_device
    check_device();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='smoke':
        from candidates.uks_freq_bfs_retry_v1.smoke import smoke
        result=smoke(a.arm,a.output)
    else:
        from candidates.uks_freq_bfs_retry_v1.worker import run
        result=run('gids_default' if a.arm=='gids' else 'digit_cpu2',a.output,'off')
    check_ready();write(a.output/'report.json',dict(result,source_sha256=verify(),cache_line_bytes=1024,raw_ssd_writes=False,
        gpu_assignment=selected,ae_transport_sha256=sha(C/'transport_manifest.json')))
if __name__=='__main__':main()
