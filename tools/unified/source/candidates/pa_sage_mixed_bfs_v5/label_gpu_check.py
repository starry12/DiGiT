"""Bit-exact GPU targets, losses, gradients and Adam steps after class packing."""
import numpy as np
from .labels import upload_labels


def run():
    import torch
    values=np.r_[np.arange(172,dtype=np.float32),np.nan]
    train=np.arange(172,dtype=np.int64)
    compact,audit=upload_labels(values,train,172,'cuda:0')
    original=torch.tensor(train,device='cuda:0')
    ids=torch.arange(171,-1,-1,device='cuda:0')
    a=torch.nn.Parameter(torch.zeros((172,172),device='cuda:0'))
    b=torch.nn.Parameter(a.detach().clone())
    oa=torch.optim.Adam([a],lr=.001,weight_decay=0)
    ob=torch.optim.Adam([b],lr=.001,weight_decay=0)
    for _ in range(20):
        targets=compact[ids].long();reference=original[ids]
        assert torch.equal(targets,reference)
        la=torch.nn.functional.cross_entropy(a,targets);lb=torch.nn.functional.cross_entropy(b,reference)
        oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        la.backward();lb.backward()
        assert torch.equal(la,lb) and torch.equal(a.grad,b.grad)
        oa.step();ob.step();assert torch.equal(a,b)
    return dict(passed=True,steps=20,targets_losses_gradients_parameters_bit_exact=True,raw_ssd_access=False)
