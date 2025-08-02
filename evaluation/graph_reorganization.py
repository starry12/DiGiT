from dataloader import IGB260MDGLDataset, OGBDGLDataset

import os
import argparse, time
import numpy as np
import torch
import os.path as osp
import pandas as pd

import dgl
from dgl.data import DGLDataset
import warnings

warnings.filterwarnings("ignore")


def count_repeated_elements(tensors):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(tensors)
    repeated_counts = torch.zeros((n, n), dtype=torch.int).to(device)

    tensors = [torch.tensor(tensor).to(device) for tensor in tensors]

    for i in range(n):
        for j in range(i + 1, n):
            # 使用集合操作计算重复元素的个数
            repeated_counts[i, j] = torch.sum(torch.isin(tensors[i], tensors[j])).item()
            repeated_counts[j, i] = repeated_counts[i, j]  # 对称矩阵

    return repeated_counts.cpu()


def graph_reorganization(g, hop, degree_threshold, top_cn):
    degree = g.in_degrees(g.nodes()).type(torch.float32)
    # 选出degree 小于 degree_threshold的节点，选出对应的节点的index
    candicate_nodes = torch.nonzero(degree < degree_threshold).squeeze()
    # 从candicate_nodes中选出in degree最大的top_cn个节点
    top_cn_nodes = torch.topk(degree[candicate_nodes], top_cn)
    indices = top_cn_nodes.indices
    indices_t = torch.tensor(indices)

    # 以这些节点为种子，进行 hop 跳的邻居采样
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(hop)
    dataloader = dgl.dataloading.DataLoader(
        g,
        indices_t,
        sampler,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    working_set = [None for i in range(len(indices))]
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        # 构造采样块（working set）
        working_set[step] = input_nodes

    nodes_total = torch.tensor([])
    for nodes in working_set:
        nodes_total = torch.cat((nodes_total, nodes))
    nodes_total = torch.unique(nodes_total)
    print("Total re-org nodes: ", len(nodes_total))

    vertex_sequence = []
    working_sets = [1 for _ in range(len(indices))]
    # 统计每个采样块的节点集合交集
    repeated_counts = count_repeated_elements(working_set)
    # print(repeated_counts)
    cur_workingsets_idx = 0
    vertex_sequence.append(cur_workingsets_idx)

    while sum(working_sets) > 0: # 贪心排序采样块顺序，使得相邻块节点重叠最大
        max_repeated = 0
        max_j_idx = -1
        for j in range(len(indices)):
            if working_sets[j] == 1:
                if repeated_counts[cur_workingsets_idx][j] >= max_repeated:
                    max_j_idx = j
        vertex_sequence.append(max_j_idx)
        working_sets[cur_workingsets_idx] = 0
        cur_workingsets_idx = max_j_idx

    feature_map = torch.full_like(g.nodes(), -1)
    storage_map = torch.full_like(g.nodes(), -1)
    flags = torch.full_like(g.nodes(), -1)
    total_tensor = torch.tensor([], dtype=torch.int)
    for i in range(len(vertex_sequence)):
        total_tensor = torch.cat((total_tensor, working_set[vertex_sequence[i]]))
    total_tensor = torch.cat((total_tensor, g.nodes()))
    cur_idx = 0
    for i in range(len(total_tensor)):
        if flags[total_tensor[i]] == -1:
            flags[total_tensor[i]] = 1

            storage_map[cur_idx] = total_tensor[i]
            feature_map[total_tensor[i]] = cur_idx
            cur_idx += 1

    return feature_map, storage_map


if __name__ == "__main__":
    from common_parser import get_homogenous_train_parser
    parser = get_homogenous_train_parser()
    args = parser.parse_args()

    device = f"cuda:" + str(args.device) if torch.cuda.is_available() else "cpu"
    if args.data == "IGB":
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
        g = g.formats("csc")
    elif args.data == "OGB":
        print("Dataset: OGB")
        dataset = OGBDGLDataset(args)
        g = dataset[0]
        g = g.formats("csc")
    else:
        g = None
        dataset = None

    hop = args.num_layers # 3层
    # 选出入度小于 8 的节点中前 100 个作为种子
    feature_map, storage_map = graph_reorganization(g, hop, 8, 100)

    origin_feat_path = None
    new_feat_path = None
    feature_map_path = None

    if args.data == "IGB": # 生成新的feature文件
        if args.dataset_size == "small":
            path = os.path.join(args.path, "small/processed/paper")
            origin_feat_path = os.path.join(path, "node_feat.npy")
            new_feat_path = os.path.join(path, "re_node_feat.npy")
            feature_map_path = os.path.join(path, "feature_map.npy")
        
        if os.path.exists(feature_map_path):
            os.remove(feature_map_path)
        if os.path.exists(new_feat_path):
            os.remove(new_feat_path)
        np.save(feature_map_path, feature_map.numpy())
        origin_feat = np.load(origin_feat_path)
        new_feat = origin_feat[storage_map]
        np.save(new_feat_path,new_feat)
        pass

    elif args.data == "OGB":
        np.save(f"papers100M_feature_map.npy", feature_map.numpy())
        origin_feat = np.load("/home/embed/OGB/papers100M-bin/raw/node_feat.npy")
        new_feat = origin_feat[storage_map]
        np.save("/home/embed/OGB/papers100M-bin/raw/re_node_feat.npy", new_feat)
        pass
    else:
        g = None
        dataset = None
