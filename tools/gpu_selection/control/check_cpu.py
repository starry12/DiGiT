"""CPU-only transport integration check; never start a native worker or GPU query."""
import argparse,importlib.util,json,os,sys,uuid,tempfile
import torch
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gpu_selection as auto_gpu
ADMIN=Path('/srv/digit-ae/admin')
ROOT=Path('/home/embed/digit')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--workflow',choices=('pa','ablation','layout','ig'),required=True);p.add_argument('--staged',type=Path);a=p.parse_args()
    admin=a.staged if a.staged else ADMIN
    sys.path.insert(0,str(ROOT))
    def forbidden(*args,**kw):raise RuntimeError('CPU check attempted GPU/process execution')
    with patch('subprocess.Popen',side_effect=forbidden),patch.object(auto_gpu,'query',side_effect=forbidden):
        if a.workflow=='pa':
            m=load(admin/'selfservice_v2/controller.py','pa_control')
        elif a.workflow=='ablation':
            sys.path.insert(0,str(ADMIN/'ablation_v1'))
            m=load(admin/'ablation_v1/controller.py','ablation_control')
            w=load(admin/'ablation_v1/ablation_worker.py','ablation_worker_check')
        elif a.workflow=='layout':
            m=load(admin/'layout_v1/controller.py','layout_control')
            w=load(admin/'layout_v1/point_runner.py','layout_worker_check')
        else:
            m=load(admin/'ig_performance_v1/runner.py','ig_runner_check')
        checks=[]
        for gpu in range(4):
            selected=dict(index=gpu,uuid='GPU-cpu-test-%d'%gpu,name='NVIDIA L40')
            auto_gpu.install_assignment(dict(selected=selected))
            assert auto_gpu.assignment()==selected
            assert auto_gpu.propagated()['CUDA_VISIBLE_DEVICES']==str(gpu)
            if a.workflow=='pa':
                policy=m.policy();assert policy['gpu']==gpu and policy['gpu_uuid']==selected['uuid']
                runtime=m.RUNTIME
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        m.RUNTIME=Path(tmp);env=m.clean_env(Path('cpu-check-job'))
                        assert env['CUDA_VISIBLE_DEVICES']==str(gpu) and env['DIGIT_AE_GPU_ASSIGNMENT']==os.environ['DIGIT_AE_GPU_ASSIGNMENT']
                finally:m.RUNTIME=runtime
            elif a.workflow=='ig':
                c=m.controller();assert len(c.jobs())==10
                assert c.environment(gpu)['CUDA_VISIBLE_DEVICES']==str(gpu)
                assert c.environment(gpu)['DIGIT_AE_GPU_ASSIGNMENT']==os.environ['DIGIT_AE_GPU_ASSIGNMENT']
                cmd=c.command(c.jobs()[0],Path('/tmp/not-executed'))
                assert cmd[1:5]==['-I','-B','-u','/srv/digit-ae/admin/gpu_selection_v1/ig_worker.py']
                import ig_telemetry
                telemetry=ig_telemetry.TrainingTelemetry('/tmp/not-executed','dummy')
                assert telemetry.gpu==gpu
                from candidates.ig_sage_host_telemetry_v1 import telemetry_review
                assert telemetry_review.review.__module__=='ig_telemetry_review'
            checks.append(gpu)
        import torch
        assert not torch.cuda.is_initialized()
    print(json.dumps(dict(passed=True,workflow=a.workflow,simulated_gpus=checks,cuda_initialized=False,training_started=False,transport_sha256=auto_gpu.transport())))
if __name__=='__main__':main()
