"""CPU reference: random groups first, then random ordinary occurrences."""
import numpy as np

def select(candidates,budget,device,*,num_nodes,choices=None):
    import torch
    from digit.outer_selection import SelectedUnits
    if choices is not None:
        raise ValueError('Weighted-policy replay streams are not valid for group-first selection')
    if budget < -1:raise ValueError('budget must be nonnegative or -1')
    ids=candidates.graph_ids
    if budget==-1:return SelectedUnits(np.arange(len(ids),dtype=np.int64),None,'all')
    remaining=int(budget);available=list(range(len(ids)));chosen=[]
    while remaining>0 and available:
        groups=[i for i in available if ids[i]>=num_nodes and candidates.costs[i]<=remaining]
        eligible=groups if groups else [i for i in available if ids[i]<num_nodes and candidates.costs[i]<=remaining]
        if not eligible:break
        # Uniform within the selected class; group size is constant for a layout.
        at=int(torch.randint(len(eligible),(1,),device=device).item())
        index=eligible[at];chosen.append(index);available.remove(index)
        remaining-=int(candidates.costs[index])
    return SelectedUnits(np.asarray(chosen,dtype=np.int64),remaining,'budget' if remaining==0 else 'no_fit')
