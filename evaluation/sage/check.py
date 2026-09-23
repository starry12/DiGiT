"""Hash selected graph/source inputs and the g2 artifact; never open the raw SSD."""
import argparse,os,time
from pathlib import Path
from evaluation.sage.common import *
from training.sage.run import input_binding

def check(output,gpu=None):
    execution=verify();output=output_path(output);output.mkdir(parents=True);start=time.time()
    state=dict(passed=False,complete=False,stage='hashing_selected_inputs',raw_ssd_access=False,submission_sha256=execution)
    write(output/'status.json',state)
    try:
        binding=input_binding(output,verify_worker());print('Selected graph/source hashes passed; checking full g2 artifact',flush=True)
        state.update(stage='checking_g2_artifact');write(output/'status.json',state);setup()
        from digit.artifacts import load_artifact_bundle
        from digit.ssd_payload import validate_bundle_verify_receipt
        from training.sage.admission import estimate
        p=cfg();base=ROOT/p['base_layout'];bundle=load_artifact_bundle(base/'final/bundle',validation_mode='fast');ssd=read(base/'ssd_ready.json')
        validate_bundle_verify_receipt(ssd['verify_receipt'],bundle,device_offset_bytes=ssd['offset'])
        budget=estimate();live=None
        if gpu is not None:
            import torch
            from digit.gpu_admission import check_live
            require(torch.cuda.device_count()==1,'Expected exactly one visible GPU')
            require(torch.cuda.get_device_capability()==(8,9),'Frozen native sampler targets sm89; a new architecture needs a new validated build')
            live=check_live(budget);require(live['passed'],'Live resource admission failed')
        state.update(passed=True,complete=True,stage='complete',filesystem_inputs_ready=True,budget=budget,live_admission=live,
                     input_binding_sha256=sha(output/'inputs.json'),seconds=time.time()-start,
                     scope='Selected graph/source and full g2 filesystem validation; SSD receipts checked, no fresh raw SSD readback or native training')
    except BaseException as exc:state.update(stage='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:write(output/'status.json',state)
    return state

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    value=check(a.output,a.gpu);print(__import__('json').dumps(value,indent=2))
if __name__=='__main__':main()
