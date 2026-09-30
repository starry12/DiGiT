"""Device-routing adapter; frozen IG sampling/model/optimizer remain unchanged."""
import sys
sys.path.insert(0,'/home/embed/digit')
sys.path.insert(0,'/srv/digit-ae/admin/gpu_selection_v1')
import gpu_selection as auto_gpu

def main():
    auto_gpu.assignment();auto_gpu.transport();auto_gpu.idle()
    from candidates.ig_sage_host_telemetry_v1 import telemetry,worker
    from ig_telemetry import TrainingTelemetry
    telemetry.TrainingTelemetry=TrainingTelemetry
    worker.main()

if __name__=='__main__':main()
