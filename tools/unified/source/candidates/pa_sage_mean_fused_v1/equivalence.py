"""Actual 1024-root dense capture with dynamic graph sizes, tails and upstream grads."""
import torch
import dgl
from .common import require
from .model import MeanGraphSAGE, prepare_model
from candidates.pa_sage_512b_overlap_v1.gpu_tests import _cpu_tree, _compare_trees, _tree_digest


def make_batch(rows, index):
    # A CPU generator leaves the training/dropout RNG sequence untouched.
    generator = torch.Generator().manual_seed(100 + index)
    sizes = [rows + 400 + index, rows + 250, rows + 100, rows]
    blocks = []
    for ns, nd in zip(sizes, sizes[1:]):
        dst = torch.arange(1, nd, dtype=torch.int32).repeat_interleave(5)
        src = torch.randint(ns, (len(dst),), generator=generator, dtype=torch.int32)
        # Node zero has no incoming edges; two duplicate source/destination edges.
        if len(src) >= 2:
            src[1] = src[0]
        blocks.append(dgl.create_block((src.cuda(), dst.cuda()), num_src_nodes=ns,
                                      num_dst_nodes=nd, idtype=torch.int32, device='cuda:0'))
    features = torch.randn(sizes[0], 128, generator=generator).cuda()
    labels = torch.randint(172, (rows,), generator=generator).cuda()
    return blocks, features, labels


def model_equivalence():
    sizes = [1024, 1024, 907, 1024, 1, 1024, 907, 1024]
    batches = [make_batch(n, i) for i, n in enumerate(sizes)]
    results = {}; baseline = None
    for mode in ('legacy', 'fused'):
        torch.manual_seed(37); torch.cuda.manual_seed_all(37)
        model = MeanGraphSAGE(128, 128, 172, 3, .2,mean_mode=mode).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.001)
        initial_keys = list(model.state_dict())
        preparation = prepare_model(model, optimizer, 'graph')
        require(list(model.state_dict()) == initial_keys, 'State paths changed')
        grads, losses, predictions, feature_gradients, rng = [], [], [], [], []
        for blocks, x, labels in batches:
            x = x.detach().clone().requires_grad_()
            rng.append(torch.cuda.get_rng_state().clone())
            prediction = model(blocks, x)
            loss = torch.nn.functional.cross_entropy(prediction, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Copy before next replay: graph outputs and gradients use static storage.
            grads.append(_cpu_tree([p.grad for p in model.parameters()]))
            feature_gradients.append(x.grad.detach().cpu().clone())
            predictions.append(prediction.detach().cpu().clone())
            losses.append(loss.detach().cpu().clone())
            optimizer.step()
        torch.cuda.synchronize()
        dense = model.dense_summary()
        require(dense['graph_forwards'] == 5, 'Full-shape graph replay count differs')
        require(dense['eager_forwards'] == 3, 'Tail fallback differs')
        state = dict(model=_cpu_tree(model.state_dict()), adam=_cpu_tree(optimizer.state_dict()),
                     gradients=grads, input_gradients=feature_gradients, predictions=predictions,
                     losses=losses, rng=rng)
        compare = None if baseline is None else _compare_trees(baseline, state)
        if baseline is None: baseline = state
        else: require(compare['bit_exact'],'Full-shape model/gradient/Adam bits changed')
        # No-grad/eval must never replay a training graph, including full shapes.
        before = dense['graph_forwards']
        model.eval()
        with torch.no_grad():
            prediction = model(batches[0][0], batches[0][1])
            require(torch.isfinite(prediction).all().item(), 'Nonfinite eval fallback')
        require(model.dense_summary()['graph_forwards'] == before, 'Replayed captured training graph in eval')
        results[mode] = dict(preparation=preparation, training_counts=dense,
            final_model_sha256=_tree_digest(state['model']), final_adam_sha256=_tree_digest(state['adam']),
            comparison=compare, eval_fallback_passed=True)
        del model, optimizer
        torch.cuda.empty_cache()
    return dict(passed=True, model_updates=len(sizes)*2, rows=sizes, results=results,
        variable_graph_shapes=True, zero_degree_and_duplicate_edges=True,
        upstream_feature_and_parameter_gradients_compared=True, single_active_training_forward=True,
        performance_evidence=False)
