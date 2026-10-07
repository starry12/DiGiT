"""Device-routing adapter; frozen IG sampling/model/optimizer remain unchanged."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_context
model_context.activate()
import sys
sys.path.insert(0,'/srv/digit-ae/admin/identity_v1')
import stable_identity
_identity_registry=stable_identity.install('IG')
import sys
sys.path.insert(0,'/home/embed/digit')
sys.path.insert(0,'/srv/digit-ae/admin/gpu_selection_v1')
import gpu_selection as auto_gpu

def main():
    auto_gpu.assignment();auto_gpu.transport();auto_gpu.idle()
    from candidates.ig_sage_host_telemetry_v1 import telemetry
    import ig_native_worker as worker
    from ig_telemetry import TrainingTelemetry
    telemetry.TrainingTelemetry=TrainingTelemetry
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    worker.main()

if __name__=='__main__':main()
