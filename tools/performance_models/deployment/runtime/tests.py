"""Numerical model oracle, optimizer ownership, and model evidence separation."""
import copy
import unittest
import torch
import dgl
from .models import DIMENSIONS, contract, make_model, cpu_check, identity, require_identity, model_memory


def blocks():
    result = []
    for src, dst in ((8, 6), (6, 4), (4, 2)):
        # Nonuniform degrees, duplicate edges, and destination-prefix nodes.
        u = torch.tensor(list(range(dst)) + [src-1, src-2, 0, 0])
        v = torch.tensor(list(range(dst)) + [0, 1, 1, 1])
        result.append(dgl.create_block((u, v), num_src_nodes=src, num_dst_nodes=dst))
    return result


def oracle(net, bs, x, name):
    """Independent dense equations; no DGL convolution or message passing."""
    for i, (layer, block) in enumerate(zip(net.layers, bs)):
        u, v = block.edges()
        count = block.num_dst_nodes()
        if name == 'gcn':
            norm_src = torch.bincount(u, minlength=len(x)).clamp(min=1).to(x.dtype).rsqrt()
            norm_dst = torch.bincount(v, minlength=count).clamp(min=1).to(x.dtype).rsqrt()
            messages = (x * norm_src[:, None]) @ layer.weight
            out = torch.zeros(count, messages.shape[1], dtype=x.dtype)
            out.index_add_(0, v, messages[u])
            x = out * norm_dst[:, None] + layer.bias
        else:
            projected = layer.fc(x).reshape(len(x), layer._num_heads, layer._out_feats)
            left = (projected * layer.attn_l).sum(-1)
            right = (projected[:count] * layer.attn_r).sum(-1)
            scores = torch.nn.functional.leaky_relu(left[u] + right[v], .2)
            out = []
            for dst in range(count):
                pick = v == dst
                weights = scores[pick].softmax(dim=0)
                out.append((weights[..., None] * projected[u[pick]]).sum(0))
            x = torch.stack(out) + layer.bias.reshape(layer._num_heads, layer._out_feats)
            x = x.flatten(1) if i < 2 else x.mean(1)
        if i < 2:
            x = torch.relu(x)
    return x


class Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_eight_cpu_training_paths(self):
        for ds in DIMENSIONS:
            for name in ('gcn', 'gat'):
                with self.subTest(dataset=ds, model=name):
                    self.assertTrue(cpu_check(ds, name)['passed'])

    def test_forward_and_gradient_oracle(self):
        for ds in DIMENSIONS:
            for name in ('gcn', 'gat'):
                with self.subTest(dataset=ds, model=name):
                    torch.random.default_generator.manual_seed(71)
                    a = make_model(ds, name, dtype=torch.float64).eval()
                    b = copy.deepcopy(a)
                    x = torch.randn(8, DIMENSIONS[ds], dtype=torch.float64)
                    bs = blocks()
                    actual, expected = a(bs, x), oracle(b, bs, x, name)
                    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
                    target = torch.tensor([2, 18])
                    for net, pred in ((a, actual), (b, expected)):
                        torch.nn.functional.cross_entropy(pred, target).backward()
                    for p, q in zip(a.parameters(), b.parameters()):
                        torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-10)

    def test_block_contract_and_zero_degree(self):
        for name in ('gcn', 'gat'):
            model = make_model('UKL', name)
            x = torch.ones(8, 128)
            with self.assertRaises(ValueError):
                model(blocks()[:2], x)
            with self.assertRaises(ValueError):
                model(blocks(), x[:, :127])
            zero = dgl.create_block((torch.tensor([0]), torch.tensor([0])), num_src_nodes=8, num_dst_nodes=6)
            with self.assertRaises(dgl.DGLError):
                model([zero, *blocks()[1:]], x)

    def test_evidence_cannot_cross_models_or_datasets(self):
        for ds in DIMENSIONS:
            for name in ('gcn', 'gat'):
                expected = identity(ds, name, 'a'*64)
                require_identity({'model_identity': expected}, expected)
                for wrong in ({}, {'model_identity': identity(ds, name, 'b'*64)},
                              {'model_identity': identity(ds, 'gat' if name=='gcn' else 'gcn', 'a'*64)},
                              {'model_identity': dict(expected, dataset='PA')}):
                    with self.assertRaises(RuntimeError):
                        require_identity(wrong, expected)

    def test_budget_model_contract(self):
        for ds in DIMENSIONS:
            for name in ('gcn', 'gat'):
                budget = model_memory(ds, name)
                self.assertEqual(budget['blocks'][0]['src'], 405504)
                self.assertGreater(budget['bytes'], sum(budget['components'].values()))
                self.assertFalse(budget['measured'])
        self.assertGreater(model_memory('UKL', 'gat')['bytes'], 2*2**30)

    def test_identical_pair_initialization(self):
        for ds in DIMENSIONS:
            for name in ('gcn', 'gat'):
                torch.random.default_generator.manual_seed(0)
                a = make_model(ds, name)
                torch.random.default_generator.manual_seed(0)
                b = make_model(ds, name)
                for p, q in zip(a.parameters(), b.parameters()):
                    self.assertTrue(torch.equal(p, q))

    def test_probe_acceptance_rejects_missing_and_cross_model_evidence(self):
        import math
        from .probe import validate
        for ds in ('UKL','CL'):
            peak = 1024**3
            receipt = dict(passed=True, dataset=ds, model='gat', blocks=model_memory(ds,'gat')['blocks'],
                           updates=4, finite=True, raw_ssd_access=False, peak_reserved_bytes=peak,
                           model_allowance_bytes=math.ceil(peak*1.25)+256*2**20, probe_limit_bytes=2*2**30)
            self.assertTrue(validate(receipt, ds, 'gat'))
            for key, value in [('passed',False),('model','gcn'),('dataset','PA'),('updates',3),
                               ('raw_ssd_access',True),('probe_limit_bytes',peak),('model_allowance_bytes',peak)]:
                self.assertFalse(validate(dict(receipt,**{key:value}),ds,'gat'))
            self.assertFalse(validate({},ds,'gat'))

    def test_checkpointed_training_matches_historical_gat(self):
        from ae.igb.models import GAT
        for ds in DIMENSIONS:
            torch.manual_seed(9)
            a=make_model(ds,'gat',dtype=torch.float64).train()
            b=GAT(**contract(ds,'gat')['config']).double().train()
            b.load_state_dict(a.state_dict())
            x=torch.randn(8,DIMENSIONS[ds],dtype=torch.float64)
            target=torch.tensor([2,18]);bs=blocks()
            oa=torch.optim.Adam(a.parameters(),lr=.001,weight_decay=.001)
            ob=torch.optim.Adam(b.parameters(),lr=.001,weight_decay=.001)
            for step in range(2):
                losses=[]
                for net,opt in ((a,oa),(b,ob)):
                    torch.manual_seed(80+step)
                    pred=net(bs,x);loss=torch.nn.functional.cross_entropy(pred,target)
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(loss.detach())
                torch.testing.assert_close(losses[0],losses[1],rtol=1e-8,atol=1e-9)
                for p,q in zip(a.parameters(),b.parameters()):
                    torch.testing.assert_close(p,q,rtol=1e-7,atol=1e-8)

    def test_native_warmup_gate_requires_actual_updates(self):
        from .gates import WarmupGate,validate
        net=make_model('UKL','gcn');opt=torch.optim.Adam(net.parameters(),lr=.001)
        gate=WarmupGate(net,opt)
        with self.assertRaises(RuntimeError):gate.require()
        for i in range(4):
            loss=torch.nn.functional.cross_entropy(net(blocks(),torch.ones(8,128)),torch.tensor([0,18]))
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            gate.observe(loss,i+1,i+1)
        result=gate.require();self.assertTrue(validate(result))
        for key,value in [('updates',3),('feature_checks',0),('optimizer_steps_verified',False),
                          ('final_model_sha256',result['initial_model_sha256'])]:
            self.assertFalse(validate(dict(result,**{key:value})))


if __name__ == '__main__':
    unittest.main()
