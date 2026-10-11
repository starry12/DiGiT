"""Launch only full filesystem CSC preparation; no GPU/SSD training or hot-set substitution."""
import argparse,json,os,shlex,shutil,subprocess
from pathlib import Path
from .transpose import ROOT,OUT,identity,sha
OUTPUT=Path('/mnt/n0/digit/ukl_sparse_overlay_v3/full_csc_v1')
SOCKET='/tmp/digit-ukl-overlay-tmux.sock';SESSION='ukl-full-csc-v3'
def main(execute=False):
 command=['/usr/bin/nice','-n','19','/usr/bin/ionice','-c','3','/home/embed/miniconda3/envs/gids/bin/python','-B','-u','-m','candidates.ukl_sparse_overlay_v3.transpose','--output',str(OUTPUT),'--full']
 value=dict(command=command,output=str(OUTPUT),tmux_socket=SOCKET,session=SESSION,execute=execute,gpu=False,raw_ssd=False,automatic_next_stage='gather and output hashing only; no training')
 print(json.dumps(value,indent=2),flush=True)
 if not execute:return
 for name in ('gpu','training','transpose','group'):
  receipt=json.loads((OUT/'checks'/(name+'.json')).read_text())
  if not receipt.get('passed'):raise RuntimeError('Check not passed '+name)
 pilot=json.loads(Path('/mnt/n0/digit/ukl_sparse_overlay_v3/pilot_100000/status.json').read_text())
 if pilot['stage']!='prefix_complete' or pilot['scatter']['owners']!=100000:raise RuntimeError('Real pilot incomplete')
 for path,v in pilot['source_identity'].items():
  if identity(Path(path))!=v:raise RuntimeError('Source identity changed')
 if sha(OUT/'build/transpose')!=pilot['transpose_binary_sha256']:raise RuntimeError('Transpose changed since real pilot')
 if OUTPUT.exists():raise RuntimeError('Preserve existing full build; inspect its status instead')
 if shutil.disk_usage(OUTPUT.parent).free<700*2**30:raise RuntimeError('Need 700 GiB free')
 mem=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
 if mem<32*2**30:raise RuntimeError('Need 32 GiB available for preparation')
 log=OUT/'full_build.log'
 if log.exists():raise RuntimeError('Preserve launch log')
 shell='exec '+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1'
 subprocess.run(['/usr/bin/tmux','-S',SOCKET,'new-session','-d','-s',SESSION,'-c',str(ROOT),shell],check=True)
 (OUT/'launch.json').write_text(json.dumps(value,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');main(p.parse_args().execute)
