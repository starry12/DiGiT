"""Three tiny updates with the selected PA models and optimizers, without GPU/SSD access."""
import argparse,json,os,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',DGLBACKEND='pytorch',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.sage.common import setup,cfg
setup()
import torch,dgl
from ae.papers.runtime.models import SAGE
from training.gcn.model import make_model as gcn
from training.gat.model import make_model as gat

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1);rows=[];n=8;u=torch.arange(n).repeat_interleave(n);v=torch.arange(n).repeat(n)
 blocks=[dgl.create_block((u,v),num_src_nodes=n,num_dst_nodes=n) for _ in range(3)]
 for name in ('sage','gcn','gat'):
  config=json.loads((ROOT/'training'/name/'protocol.json').read_text());torch.manual_seed(config['seed'])
  model=SAGE(in_feats=128,h_feats=config['hidden'],num_classes=config['classes'],num_layers=config['layers'],dropout=config['dropout']) if name=='sage' else {'gcn':gcn,'gat':gat}[name]()
  options=dict(config['optimizer']['kwargs']);options['betas']=tuple(options['betas']);opt=torch.optim.Adam(model.parameters(),**options)
  x=torch.randn(n,128);y=torch.arange(n)%172;losses=[]
  for step in range(3):
   before=[p.detach().clone() for p in model.parameters()];pred=model(blocks,x)
   if pred.shape!=(n,172):raise RuntimeError('Wrong output shape')
   opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(pred,y);loss.backward()
   if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradients')
   opt.step()
   if not any(not torch.equal(a,b) for a,b in zip(before,model.parameters())):raise RuntimeError('No update')
   losses.append(loss.item())
  rows.append(dict(model=name,passed=True,updates=3,losses=losses,optimizer=config['optimizer']))
 report=dict(passed=True,cpu_only=True,native_acceptance=False,rows=rows)
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
