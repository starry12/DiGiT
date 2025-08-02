import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
from dataloader import IGB260MDGLDataset, OGBDGLDataset, UKSDGLDataset
import csv 
import warnings
import os
import numpy as np
import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc

import GIDS
from GIDS import GIDS_DGLDataLoader, get_sample_record, get_fetch_feature_record

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from wq.custom_sample import multi_layer_custom_sampling

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")


@nvtx.annotate("fetch_data_chunk()", color="blue")
def fetch_data_chunk(test, out_t, page_size, stream_id):
    test.fetch_from_backing_memory_chunk(out_t.data_ptr(), page_size, stream_id)



def print_times(transfer_time, train_time, e2e_time):
    print("transfer time: ", transfer_time)
    print("train time: ", train_time)
    print("e2e time: ", e2e_time)

def track_acc_GIDS(g, args, device, label_array=None):

    GIDS_Loader = None
    GIDS_Loader = GIDS.GIDS(
        page_size = args.page_size,
        off = args.offset,
        num_ele = args.num_ele,
        num_ssd = args.num_ssd,
        cache_size = args.cache_size,
        window_buffer = args.window_buffer,
        wb_size = args.wb_size,
        accumulator_flag = args.accumulator,
        cache_dim = args.cache_dim,
        ssd_list = [0],
        graph_reorganize = args.graph_reorganize,
        feature_map = args.feat_map,
        graph_record=args.save_input_nodes
    )
    dim = args.emb_size

    if(args.accumulator):
        GIDS_Loader.set_required_storage_access(args.bw, args.l_ssd, args.l_system, args.num_ssd, args.peak_percent)


    if(args.cpu_buffer):
        start_time = time.time()
        num_nodes = g.number_of_nodes()
        print(f"Number of Nodes: {num_nodes}")
        num_pinned_nodes = int(num_nodes * args.cpu_buffer_percent)
        print(f"Number of CPU Pinned Nodes: {num_pinned_nodes}")
        GIDS_Loader.cpu_backing_buffer(dim, num_pinned_nodes)
        print("CPU Buffer Created, dim: ", dim)
        print(type(g))  # 输出示例：<class 'dgl.heterograph.DGLGraph'>
        pr_ten = torch.load(args.pin_file)
        GIDS_Loader.set_cpu_buffer(pr_ten, num_pinned_nodes)
        GIDS_Loader.print_stats()
        print("############################################################################")
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"代码块执行时间: {elapsed:.6f} 秒")


    sampler = dgl.dataloading.MultiLayerNeighborSampler(
               [int(fanout) for fanout in args.fan_out.split(',')]
               )
    
    '''
    # 加载
    group_ptr = np.load('group_ptr.npy')      # shape (num_nodes+1,)
    group_idxs = np.load('group_idxs.npy')    # shape (num_groups,)

    device = g.device
    # 转为 GPU tensor
    group_ptr_t  = torch.from_numpy(group_ptr).long().to(device)
    group_idxs_t = torch.from_numpy(group_idxs).long().to(device)

    sampler = dgl.dataloading.GroupNeighborSampler(
        [int(f) for f in args.fan_out.split(',')],
        use_group_pre=True,
        group_ptr=group_ptr_t,
        group_idxs=group_idxs_t,
        group_size=2,
        edge_dir='in',  # 同原
        prob=None,
        replace=False,
        output_device=args.device,
    )
    '''

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    # 打印train_nid第一个是false的位置
    print(torch.nonzero(g.ndata['train_mask'], as_tuple=True))
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    #in_feats = g.ndata['features'].shape[1]
    #print("in_feats: ", in_feats) 
    #OGB
    in_feats = args.emb_size
    print(train_nid.shape, val_nid.shape, test_nid.shape)

    train_dataloader = GIDS_DGLDataLoader(
        g,
        train_nid,
        sampler,
        args.batch_size,
        dim,
        GIDS_Loader,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_alternate_streams=False
        )


    val_dataloader = dgl.dataloading.DataLoader(
        g, val_nid, sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, test_nid, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)

    if args.model_type == 'gcn':
        model = GCN(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'sage':
        model = SAGE(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'gat':
        model = GAT(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers, args.num_heads).to(device)

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay
        )
    train_time_jc = 0
    warm_up_iter = 100
    transfer_time = 0
    train_time = 0
    e2e_time = 0
    total_batch = int((len(train_nid) + args.batch_size - 1) / args.batch_size)
    print("Total Batch: ", total_batch)
    epoch_tqdm = tqdm.trange(args.epochs, position=0, leave=True, ascii=True, desc="Epoch")
    batch_tqdm = tqdm.trange(total_batch, position=1, leave=False, ascii=True, desc="Batch")
    test_tqdm = tqdm.trange(int((len(test_nid) + args.batch_size - 1) / args.batch_size), position=2, leave=False, ascii=True, desc="Test")
    #print("blocks_record load")
    #blocks_record = np.load("/home/embed/Downloads/igb_large/Blocks_IGB_large.npy", allow_pickle=True)
    #print("blocks_record load done")

    # Setup is Done
    #num_nodes = 269_364_174
    #num_nodes = 100_000_000
    # 用 uint16（int16）约可表示 0～65535 次累加，全程约 514 MiB
    #freq = torch.zeros(num_nodes, dtype=torch.int16, pin_memory=True)
    #assert freq.device.type == "cpu"


    for epoch in range(args.epochs):
        print("start epoch")
        epoch_start = time.time()
        epoch_loss = 0
        train_acc = 0
        model.train()

        #batch_input_time = 0
        #train_time = 0
        #transfer_time = 0
        #e2e_time = 0
        e2e_time_start = time.time()
        # epoch_nodes = []
        # epoch_nodes_size = []
        epoch_io_stats = []
        block_relationships = []

        # 在DGL中，blocks 是从图数据中采样得到的一系列子图，这些子图代表了训练过程中需要处理的局部图结构。
        # 每个 block 对应图神经网络中的一层。
        # 例如，如果你有一个两层的图卷积网络，那么在每次训练步骤中，你可能会得到两个blocks：
            # 第一个 block 包含第一层需要的输入节点和边
            # 第二个 block 包含第二层需要的输入节点和边

        log_interval = 100          # 每 N 个批次打印一次
        num_batches  = 0            # 当前 epoch 的批次数

        for step, (input_nodes, seeds, blocks, ret) in enumerate(train_dataloader):
            #if step == 1:
                #break
            #fanouts = [10, 5, 5]
            '''
            id_list = [blk.srcdata[dgl.NID].cpu() for blk in blocks]  # 每个都是 CPU Tensor
            all_ids = torch.cat(id_list, dim=0)                         # 合并，CPU 上的 1D 张量
            unique_ids = torch.unique(all_ids)                          # 去重，CPU 张量

            # 再次确认
            assert unique_ids.device.type == "cpu"
            assert freq.device.type == "cpu"

            # “ones” 也要在 CPU
            ones_cpu = torch.ones(unique_ids.size(), dtype=freq.dtype, device=freq.device)
            freq.index_add_(0, unique_ids, ones_cpu)
            '''
            #print(blocks)
            #for i, block in enumerate(blocks):
                #print(f"Block {i}:")
                #print(block)
            
            '''

            blocks = multi_layer_custom_sampling(
                g,
                seeds,
                fanouts,
                replace=False,
                edge_dir='in',  # 采样入边；若需要采样出边，则设置为 'out'
                prob=None,    # 若需要使用边权重进行非均匀采样，可以传入边权重字段名
                pre_srcs_file='pre_srcs.npy',
                pre_groups_file='pre_groups.npy'
            )
            '''
            torch.set_printoptions(threshold=1024)
           # print(seeds)

            if args.save_blocks:
                for i, block in enumerate(blocks):
                    # Get source and destination nodes for this block
                    src_nodes = block.srcdata[dgl.NID].cpu().numpy()
                    dst_nodes = block.dstdata[dgl.NID].cpu().numpy()
                    # Get edge information to see neighbor connections
                    u, v = block.edges()
                    u = u.cpu().numpy()  # Source node indices within the block
                    v = v.cpu().numpy()  # Destination node indices within the block
                    
                    # Map local indices to global node IDs
                    global_u = src_nodes[u]  # Convert to global node IDs
                    global_v = dst_nodes[v]  # Convert to global node IDs
                    
                    # Create a dictionary mapping each destination node to its sampled neighbors
                    neighbor_map = {}
                    for idx in range(len(u)):
                        dst_node = global_v[idx]
                        src_node = global_u[idx]
                        if dst_node not in neighbor_map:
                            neighbor_map[dst_node] = []
                        neighbor_map[dst_node].append(src_node)
                    
                    # Save mapping between layers
                    block_relationships.append({
                        'layer': i,
                        'src_nodes': src_nodes,
                        'dst_nodes': dst_nodes,
                        'neighbor_map': neighbor_map,
                        'step': step,
                        'epoch': epoch
                    })
            
            #blocks = [b for b in blocks_record 
                   #if b['step'] == step]

            # Features are fetched by the baseline GIDS dataloader in ret
            # ret 是一个 单个 torch.Tensor
            # 代表了 当前 batch 所需的全部节点特征 inputs
            batch_inputs = ret
            #print(batch_inputs)
            

            transfer_start = time.time() 
            batch_labels = blocks[-1].dstdata['labels']


            #print("blocks: ", blocks)

            # 将采样完的每层子图结构传输给GPU
            blocks = [block.int().to(device) for block in blocks]

            batch_labels = batch_labels.to(device)
            transfer_time += time.time()  - transfer_start

            '''
            # 1. 从最外层 block 拿到 src 节点 global id（GPU 张量）
            outer_src_nids = blocks[0].srcdata[dgl.NID]

            # 2. 把索引挪到 CPU，去索引原图的 CPU 特征
            batch_inputs = g.ndata['feat'][outer_src_nids.cpu()]

            # 3. 再把特征搬到 GPU
            batch_inputs = batch_inputs.to(device)

            '''
 
            # Model Training Stage
            # train start
            train_start = time.time()
            start_time = time.perf_counter_ns()

            batch_pred = model(blocks, batch_inputs)

            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
            num_batches += 1
            end_time = time.perf_counter_ns()
            train_time += time.time() - train_start
            train_time_jc += end_time - start_time
            #if (step + 1) % log_interval == 0:
                #print(f"Epoch {epoch:03d} | Step {step+1:05d} | "
                    #f"Batch Loss {loss.item():.4f} | "
                    #f"Avg Loss {epoch_loss / num_batches:.4f}")

            # train end
            batch_tqdm.update()
            if args.save_input_nodes:
                GIDS_Loader.save_epoch_nodes(args=args)
        
            

            '''
            #print("step: ", step)
            #1178
            #if(step < 10):
                #print("step: ", step)
                #train_dataloader.print_stats()
                #train_dataloader.print_timer()
                #print("transfer time: ", transfer_time)
                #print("train time: ", train_time)
          
            #if(step == warm_up_iter + 100):
                #print("Performance for 100 iteration after 1000 iteration")
                #e2e_time += time.time() - e2e_time_start 
                #train_dataloader.print_stats()
                #train_dataloader.print_timer()
                #print_times(transfer_time, train_time, e2e_time)
             
                #batch_input_time = 0
                #transfer_time = 0
                #train_time = 0
                #e2e_time = 0
                
                #Just testing 100 iterations remove the next line if you do not want to halt
                #return None
        
            # np.save(f"{args.path}/IO-STAT_{args.data}_{args.dataset_size}_{epoch}.npy", 
            #     np.concatenate(epoch_io_stats))
            # np.save(f"{args.path}/Batch-Nodes_{args.data}_{args.dataset_size}_{epoch}.npy", 
            #         np.concatenate(epoch_nodes,dtype=np.int64))
            # np.save(f"{args.path}/Batch-Nodes-size_{args.data}_{args.dataset_size}_{epoch}.npy", 
            #         np.array(epoch_nodes_size),dtype=np.int64)
            # if step == 1000:
            #     break
            '''

        print(f"Epoch {epoch:03d} completed. "
            f"Average Loss: {epoch_loss / num_batches:.4f}")
        
        print("Performance for one epoch")
        #torch.save(all_unique_ids, "all_unique_ids_int64.pt")
        #print(f"Saved all_unique_ids (dtype=int64, len={len(all_unique_ids)}) to 'all_unique_ids_int64.pt'")
        #torch.save(freq, "node_freq_papers100M.pt")
        #print(f"Finished training. Node-frequency tensor saved to 'node_freq.pt'.")
        e2e_time += time.time() - e2e_time_start 
        train_dataloader.print_stats()
        
        train_dataloader.print_timer()
        print_times(transfer_time, train_time, e2e_time)
        sample_cnt, sample_time = get_sample_record()
        fetch_feature_cnt, fetch_feature_time = get_fetch_feature_record()
        total_time_ns = sample_time + fetch_feature_time + train_time_jc
        total_time_sec = total_time_ns / 1e9  # 转换为秒
        sample_time_sec = sample_time / 1e9  # 转换为秒
        fetch_feature_time_sec = fetch_feature_time / 1e9  # 转换为秒
        train_time_jc_sec = train_time_jc / 1e9  # 转换为秒
        print(f"Sample Count: {sample_cnt}, Sample Time(ns): {sample_time}, Sample Time(s): {sample_time_sec:.6f}")
        print(f"Fetch Feature Count: {fetch_feature_cnt}, Fetch Feature Time(ns): {fetch_feature_time}, Fetch Feature Time(s): {fetch_feature_time_sec:.6f}")
        print(f"Train Time(ns): {train_time_jc}, Train Time(s): {train_time_jc_sec:.6f}")
        print(f"Total Time(ns): {total_time_ns}, Total Time(s): {total_time_sec:.6f}")

        epoch_tqdm.update()
        if args.save_blocks:
            np.save(f"{args.path}/Blocks_{args.data}_{args.dataset_size}.npy", block_relationships)
        if args.save_input_nodes:
            GIDS_Loader.save_epoch_nodes(args=args, flag = True)
        #sample_time = e2e_time - transfer_time - train_time
        #print("sample_time: ", sample_time)
             
        #batch_input_time = 0
        #transfer_time = 0
        #train_time = 0
        #e2e_time = 0

    
    # Evaluation
    print("Evaluation")
    model.eval()
    predictions = []
    labels = []
    print("Using device test:", device)
    with torch.no_grad():
        for step, (_, _, blocks) in enumerate(test_dataloader):
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata['feat']
     
            if(args.data == 'IGB'):
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())
            elif(args.data == 'OGB'):
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())
                # out_label = torch.index_select(label_array, 0, b[1]).flatten()
                # labels.append(out_label.numpy())
            predict = model(blocks, inputs).argmax(1).cpu().numpy()
            predictions.append(predict)
            test_tqdm.update()
            

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
    print("Test Acc {:.2f}%".format(test_acc))
    
    

if __name__ == '__main__':
    from common_parser import get_homogenous_train_parser
    parser = get_homogenous_train_parser()
    args = parser.parse_args()
    print("GIDS DataLoader Setting")
    print("GIDS: ", args.GIDS)
    print("CPU Feature Buffer: ", args.cpu_buffer)
    print("Window Buffering: ", args.window_buffer)
    print("Storage Access Accumulator: ", args.accumulator)

    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    feature_map_path = None
    args.feat_map = None

    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        #print(args)
        if args.graph_reorganize == True:
            if args.dataset_size == "small":
                feature_map_path = os.path.join(args.path, "small/processed/paper","feature_map.npy")
            feature_map = np.load(feature_map_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.feat_map = torch.from_numpy(feature_map).to(device)
        
        g = dataset[0]
        g  = g.formats('csc')
        print(g)

        #node_ids = g.nodes()
        # 如果 node_ids 是一个 torch.Tensor 则可以直接使用以下方法查找最小值和最大值
        #min_node = torch.min(node_ids)
        #max_node = torch.max(node_ids)
        #print("节点 ID 范围: {} 到 {}".format(min_node.item(), max_node.item()))


    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    elif(args.data == "UKS"):
        print("Dataset: UKS")
        dataset = UKSDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
        print(g)
    else:
        g=None
        dataset=None
    
    track_acc_GIDS(g, args, device, labels)




