"""Block comparison only; no test-module import or thread-setting side effects."""
import torch

def equal_blocks(test, left, right):
    test.assertEqual(len(left), len(right))
    for a, b in zip(left, right):
        test.assertEqual(a.num_src_nodes(), b.num_src_nodes())
        test.assertEqual(a.num_dst_nodes(), b.num_dst_nodes())
        for x, y in zip(a.edges(order='eid'), b.edges(order='eid')):
            test.assertTrue(torch.equal(x.cpu(), y.cpu()))
        for x, y in ((a.srcdata, b.srcdata), (a.dstdata, b.dstdata), (a.edata, b.edata)):
            test.assertEqual(set(x), set(y))
            for k in x:
                test.assertTrue(torch.equal(x[k].cpu(), y[k].cpu()), k)

