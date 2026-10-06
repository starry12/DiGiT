"""CPU-only namespace, original input identity and native-import selftest."""
import argparse,importlib,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
sys.path.insert(0,str(ROOT))
ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=('gids','digit'),default='gids');args=ap.parse_args()
identity()
import torch
torch.set_num_threads(1)
from candidates.ukl_common_gpu_five_v27 import protocol as P
require(P.verify_manifest()==read(CONTROL/'snapshot_manifest.json')['source_manifest_sha256'],'Source manifest changed')
P.configure_stage('gids');P.require_predecessor();P.binding();P.source_device()
from candidates.ukl_common_gpu_five_v27.inputs import accepted_inputs,arm_inputs,load_window
inputs=accepted_inputs()
for arm in ('gids','digit'):
 roots,labels=load_window(arm_inputs(inputs,arm));require(roots.shape==labels.shape==(320,1024),'Window shape')
for name in ('start','worker','session','stability','numa','numa_monitor'):
 importlib.import_module('candidates.ukl_common_gpu_five_v27.'+name)
from candidates.ukl_common_gpu_five_v27.backend import prepare_dependencies
P.configure_stage(args.arm)
prepare_dependencies()
from candidates.ukl_common_gpu_five_v27.numa import require_nodes
require_nodes()
import torch
require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
import transport
args=transport.adapt_command(['/usr/bin/systemd-run','/usr/bin/python3','-B','-m','candidates.ukl_common_gpu_five_v27.probe_io_stat'])
require(any(str(SNAPSHOT)+':'+str(ROOT) in a for a in args),'Missing child snapshot mount')
print('PASS: UKL namespace, immutable dependencies, original file identities, 20+300 windows and CPU imports; no GPU/SSD run')
