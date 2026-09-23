"""GAT attention/optimizer oracle, dispatch, failed-smoke gating and evidence scope."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest import mock
import torch,dgl
from training.gat.common import ROOT,HERE,cfg,require,read,write,setup
from training.gat.model import make_model
from ae.papers.runtime.models import GAT as HistoricalGAT


def blocks(device='cpu'):
    result=[]
    for src,dst in ((7,5),(5,3),(3,2)):
        u=list(range(src))+list(range(dst))+[0,0];v=[i%dst for i in range(src)]+list(range(dst))+[0,0]
        b=dgl.create_block((torch.tensor(u),torch.tensor(v)),num_src_nodes=src,num_dst_nodes=dst).to(device)
        b.srcdata[dgl.NID]=torch.arange(src,device=device);b.dstdata[dgl.NID]=torch.arange(dst,device=device);result.append(b)
    return result

def oracle(model,bs,x):
    h=x
    for i,(layer,b) in enumerate(zip(model.layers,bs)):
        u,v=b.edges(order='eid');dst=b.num_dst_nodes();heads=layer._num_heads;width=layer._out_feats
        # Two independent linear projections reproduce the historical tuple path.
        src=torch.nn.functional.linear(h,layer.fc.weight).reshape(-1,heads,width)
        dest=torch.nn.functional.linear(h[:dst],layer.fc.weight).reshape(-1,heads,width)
        scores=torch.nn.functional.leaky_relu((src*layer.attn_l).sum(-1)[u]+(dest*layer.attn_r).sum(-1)[v],negative_slope=.2)
        # Softmax separately for each destination and head, including duplicate edges.
        h=torch.stack([(src[u[v==node]]*torch.softmax(scores[v==node],dim=0).unsqueeze(-1)).sum(0) for node in range(dst)])
        h=h+layer.bias.reshape(heads,width)
        if i<len(model.layers)-1:h=torch.nn.functional.dropout(torch.relu(h.flatten(1)),p=model.dropout.p,training=model.training)
        else:h=h.mean(1)
    return h

def math_check(device='cpu'):
    torch.manual_seed(71);model=make_model(device,dtype=torch.float64);ref=copy.deepcopy(model);bs=blocks(device)
    x=torch.randn(7,128,device=device,dtype=torch.float64);labels=torch.tensor([0,171],device=device)
    options=dict(cfg()['optimizer']['kwargs']);options['betas']=tuple(options['betas'])
    opt=torch.optim.Adam(model.parameters(),**options);other=torch.optim.Adam(ref.parameters(),**options)
    errors=[]
    for step in range(3):
        torch.manual_seed(90+step);pred=model(bs,x);loss=torch.nn.functional.cross_entropy(pred,labels);opt.zero_grad(set_to_none=True);loss.backward()
        torch.manual_seed(90+step);want=oracle(ref,bs,x);expected=torch.nn.functional.cross_entropy(want,labels);other.zero_grad(set_to_none=True);expected.backward()
        torch.testing.assert_close(pred,want,rtol=1e-8,atol=1e-10)
        errors.append(float((pred-want).abs().max()))
        for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=1e-8,atol=1e-10)
        opt.step();other.step()
        for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    return dict(passed=True,device=device,updates=3,dropout_and_weight_decay_exercised=True,max_abs_logits_error=max(errors))

class ModelChecks(unittest.TestCase):
    def test_manual_attention_gradients_and_adam(self):self.assertTrue(math_check()['passed'])
    def test_historical_gat_math_and_initialization_unchanged(self):
        torch.manual_seed(0);a=make_model();torch.manual_seed(0);b=HistoricalGAT(**cfg()['model_config'])
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for k in a.state_dict():self.assertTrue(torch.equal(a.state_dict()[k],b.state_dict()[k]))
        x=torch.randn(7,128);bs=blocks();torch.manual_seed(8);y=a(bs,x);torch.manual_seed(8);self.assertTrue(torch.equal(y,b(bs,x)))
    def test_reject_truncated_blocks_and_wrong_rows(self):
        a=make_model();bs=blocks()
        for sequence,x in ((bs[:2],torch.randn(7,128)),(bs,torch.randn(8,128)),(bs,torch.randn(7,64))):
            with self.assertRaises(RuntimeError):a(sequence,x)
    def test_zero_indegree_is_not_silently_allowed(self):
        bs=blocks();bs[0]=dgl.create_block((torch.tensor([0]),torch.tensor([0])),num_src_nodes=7,num_dst_nodes=5)
        with self.assertRaises(dgl.DGLError):make_model()(bs,torch.randn(7,128))
    def test_actual_gat_parameter_count_and_explicit_heads(self):
        m=make_model();self.assertEqual(sum(p.numel() for p in m.parameters()),685072);self.assertEqual([l._num_heads*l._out_feats for l in m.layers],[512,512,688])
    def test_protocol_uses_paper_sampling_and_gat_optimizer(self):
        from training.gat.inputs import compatibility
        self.assertTrue(compatibility()['passed']);p=cfg();self.assertEqual(p['fanouts'],[10,5,5]);self.assertEqual(p['optimizer']['kwargs']['lr'],.01);self.assertEqual(p['optimizer']['kwargs']['weight_decay'],.001)
    def test_gat_reserve_preserves_shared_budget(self):
        from training.gat.admission import estimate
        v=estimate();self.assertGreater(v['required_bytes'],v['parent_required_bytes']);self.assertEqual(v['original_csc_gpu_bytes'],0);self.assertEqual(v['exact_parameters'],685072);self.assertEqual(v['host_required_bytes'],320*2**30);self.assertFalse(v['formal_bound'])

class EntryChecks(unittest.TestCase):
    def test_plan_uses_new_gat_worker_and_smoke_before_full(self):
        from training.gat import controller as c
        jobs=c.plan('representative');self.assertEqual([(j['mode'],j['arm']) for j in jobs],[('smoke','gids'),('smoke','digit_full'),('full','gids'),('full','digit_full')])
        for j in jobs:self.assertIn('training.gat.worker',c.worker_command(j,ROOT/'results/not_started'))
    def test_each_smoke_failure_blocks_full(self):
        from training.gat import controller as c
        for fail_arm in ('gids','digit_full'):
            with self.subTest(arm=fail_arm),tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
                called=[];output=Path(td)/'fresh'
                def fail(o,j,*args):
                    called.append(j)
                    if j['arm']==fail_arm:raise RuntimeError('smoke fixture failure')
                    return {}
                def binding(o,e):write(o/'inputs.json',{});return {}
                with mock.patch.object(c,'verify',return_value='gat'),mock.patch('evaluation.gat.common.verify',return_value='entry'),mock.patch.object(c,'input_binding',side_effect=binding),mock.patch.object(c,'run_worker',side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError,'smoke fixture failure'):c.execute('representative',output,2)
                self.assertTrue(all(j['mode']=='smoke' for j in called));self.assertFalse(read(output/'status.json')['complete'])
    def test_output_cannot_overwrite_existing_evidence(self):
        from training.gat.common import output_path
        for p in (ROOT/'results',HERE,Path('/tmp/gat_escaped')):
            with self.assertRaises(RuntimeError):output_path(p)
    def test_fake_completed_output_rejected(self):
        from training.gat.summarize import load_formal
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
            p=Path(td)
            for n in ('status','launch','inputs'):write(p/(n+'.json'),{})
            with self.assertRaises((KeyError,RuntimeError)):load_formal(p)

def run_checks():
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__,fromlist=[''])))
    value=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),raw_ssd_access=False)
    require(value['passed'],'GAT CPU regressions failed');return value
if __name__=='__main__':print(json.dumps(run_checks(),indent=2))
