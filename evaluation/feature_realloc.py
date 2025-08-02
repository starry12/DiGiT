from dataloader import IGB260M

import os
import argparse, time
import numpy as np
import torch
import os.path as osp
import pandas as pd

import dgl
from dgl.data import DGLDataset
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def get_neighbor_one_hop(g):
    node_indices = g.nodes()
    degree = g.in_degrees(g.nodes()).type(torch.float32)
    print("Max in degree:", torch.max(degree).item())
    # 把自己也算上
    max_neighbors = int(torch.max(degree).item())+1
    num_nodes = len(node_indices)

    # 预先分配内存
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.DataLoader(
        g,
        node_indices,
        sampler,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=32,
    )
    neighbors_tensor = torch.empty((num_nodes * max_neighbors,), dtype=torch.long)
    neighbor_idx = torch.empty((num_nodes,), dtype=torch.long)
    index = 0
    for i, (input_nodes, output_nodes, blocks) in tqdm(enumerate(dataloader)):
        neighbors = input_nodes
        num_neighbors = len(neighbors)
        neighbors_tensor[index:index + num_neighbors] = neighbors
        neighbor_idx[i] = index
        index += num_neighbors

    # 截断未使用的部分
    neighbors_tensor = neighbors_tensor[:index]

    return neighbors_tensor, neighbor_idx
    
def get_neighbot_one_hop2(edges:np.ndarray,num_nodes:int):
    in_neighbors = [ [] for _ in range(num_nodes)]
    neighbor_lens = np.zeros(num_nodes,dtype=np.int64)
    # Add Self Loop
    for node_idx in range(num_nodes):
        in_neighbors[node_idx].append(node_idx)
        neighbor_lens[node_idx] += 1
    for edge in tqdm(edges):
        in_neighbors[edge[1]].append(edge[0])
        neighbor_lens[edge[1]] += 1
    return in_neighbors, neighbor_lens


if __name__ == "__main__":
    from common_parser import get_homogenous_train_parser
    parser = get_homogenous_train_parser()
    args = parser.parse_args()
    device = f"cuda:" + str(args.device) if torch.cuda.is_available() else "cpu"
    
    # if args.data == "IGB":
    #     print("Dataset: IGB")
    #     dataset = IGB260MDGLDataset(args)
    #     g = dataset[0]
    #     g = g.formats("csc")
    # elif args.data == "OGB":
    #     print("Dataset: OGB")
    #     dataset = OGBDGLDataset(args)
    #     g = dataset[0]
    #     g = g.formats("csc")
    # else:
    #     g = None
    #     dataset = None    
    dataset = IGB260M(root=args.path, size=args.dataset_size, in_memory=0, uva_graph=args.uva_graph, \
            classes=args.num_classes, synthetic=args.synthetic, emb_size=args.emb_size, data=args.data)
    origin_feat = dataset.paper_feat
    
    origin_feat = dataset.paper_feat
    node_edges = dataset.paper_edge
    num_nodes = dataset.num_nodes()
    print("Node edges shape:", node_edges.shape)
    
    neighbor_feat_path = None
    neighbor_len_path = None
    nodes_neighbor_path = None
    if args.data == "IGB":
        if args.dataset_size == "small":
            path = os.path.join(args.path, "small/processed/paper")

    elif args.data == "OGB":
        if args.dataset_size == "papers100M":
            path = os.path.join(args.path)
    else:
        g = None
        dataset = None
    
    neighbot_path = os.path.join(path, "neighbor/")
    if not os.path.exists(neighbot_path):
        os.makedirs(neighbot_path)
    neighbor_feat_path = os.path.join(neighbot_path, "neighbor_node_feat.npy")
    neighbor_len_path = os.path.join(neighbot_path, "neightbor_node_size.npy")
    nodes_neighbor_path = os.path.join(neighbot_path, "neightbor_node.npy")    
    
    in_neighbors,neighbor_lens = get_neighbot_one_hop2(node_edges,num_nodes)
    neighbor_np = np.concatenate(in_neighbors)
    np.save(nodes_neighbor_path,neighbor_np)
    np.save(neighbor_len_path,neighbor_lens)

    # neighbors_tensor, neighbor_idx = get_neighbor_one_hop(g)
    print(neighbor_np)
    
    new_feat = origin_feat[neighbor_np]
    np.save(neighbor_feat_path,new_feat)
