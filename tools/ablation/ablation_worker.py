"""Fixed AE launcher for a sealed candidate plus explicit host192 policy."""
import importlib, json, os, sys
from pathlib import Path
ROOT=Path('/home/embed/digit')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import ablation_host_admission as admission

def execute():
    if sys.argv[1:]==['--selftest']:
        report=admission.audit_estimates()
        import torch
        admission.require(not torch.cuda.is_initialized(),'Admission selftest initialized CUDA')
        print(json.dumps(dict(report,cuda_initialized=False,training_started=False)));return
    admission.require(os.geteuid()==0 and __debug__,'Root service without Python optimization required')
    admission.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','GPU 2 only')
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('gids','gr','ns','digit_full'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    for p in (args.output,args.binding):
        admission.require(p.is_absolute() and Path('/srv/digit-ae/ablation-results') in p.parents,'Output/binding must stay in AE results')
    version='graph_v4' if args.arm=='gr' else 'cache_v3'
    common=importlib.import_module('candidates.pa_sage_ablation_'+version+'.common');common.setup()
    admission.install(version)
    # Record the actual check rather than losing its measurements on rejection.
    gpu_admission=importlib.import_module('digit.gpu_admission');original=gpu_admission.check_live
    def check_live(plan,device=0):
        value=original(plan,device=device)
        common.write(args.output/'admission.json',value)
        print('AE admission: '+json.dumps({k:value[k] for k in ('passed','host_available_bytes','host_required_bytes','free_bytes','required_bytes','host_admission_policy')}),flush=True)
        return value
    gpu_admission.check_live=check_live
    importlib.import_module('candidates.pa_sage_ablation_'+version+'.worker').main()

if __name__=='__main__':execute()
