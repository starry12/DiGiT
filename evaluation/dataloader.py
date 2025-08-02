import argparse, time
import numpy as np
import torch
import os.path as osp
import pandas as pd

import dgl
from dgl.data import DGLDataset
import warnings
warnings.filterwarnings("ignore")


def _idx_to_mask(indices, total_samples):
        mask = torch.zeros(total_samples, dtype=torch.bool)
        mask[indices] = True
        return mask

def ogb_get_idx_split_mask(path, n):
    train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
    valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
    test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

    print("train_task: ", len(train_idx))

    train_mask = _idx_to_mask(train_idx, n)
    valid_mask = _idx_to_mask(valid_idx, n)
    test_mask = _idx_to_mask(test_idx, n)

    return train_mask, valid_mask, test_mask


class IGB260M(object): # 用来定义各个部分的文件名
    def __init__(self, root: str, size: str, in_memory: int, uva_graph: int,  \
            classes: int, synthetic: int, emb_size: int, data: str,graph_reorganize: bool = False):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes
        self.emb_size = emb_size
        self.uva_graph = uva_graph
        self.data = data
        self.graph_reorganize = graph_reorganize
        #print("test1")
    
    def num_nodes(self):
        if self.data == 'OGB':
            return 111059956
        if self.data == 'UKS':
            return 133633040

        if self.size == 'experimental':
            return 100000
        elif self.size == 'small':
            # 总节点数
            return 1000000
        elif self.size == 'medium':
            return 10000000
        elif self.size == 'large':
            return 100000000
        elif self.size == 'full':
            return 269346174

    @property
    def paper_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        # TODO: temp for bafs. large and full special case
        if self.data == 'OGB':
            path = osp.join(self.dir, 'node_feat.npy')
            if self.in_memory:
                print("in_memory")
                emb = np.load(path)
            else:
                emb = np.load(path, mmap_mode='r')

        elif self.size == 'large':
            # path = '/mnt/nvme2n1/IGBH/full/processed/paper/node_feat.npy'
            path = '/mnt/n3/igb_large/processed/paper/node_feat.npy'
            print("feature file: ", path)
           
            # path = '/mnt/nvme2n1/igb_large/processed/paper/node_feat.npy'
            if self.in_memory:
                emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024)).copy()
            else:    
                emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024))
        elif self.size == 'full':
            # path = '/mnt/nvme2n1/IGBH/full/processed/paper/node_feat.npy'
            path = '/mnt/n3/IGBH/full/processed/paper/node_feat.npy'
            print("feature file: ", path)
           
            # path = '/mnt/nvme2n1/igb_large/processed/paper/node_feat.npy'
            if self.in_memory:
                emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024)).copy()
            else:    
                emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024))
        else:
            # feature file
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
            print("feature file: ", path)
            if self.synthetic:
                emb = np.random.rand(num_nodes, 1024).astype('f')
            else:
                if self.in_memory: # 1
                    emb = np.load(path)
                else:
                    emb = np.load(path, mmap_mode='r')

        return emb

    @property
    def paper_label(self) -> np.ndarray:
        if(self.data == 'OGB'):
            path = osp.join(self.dir, 'node_label.npy')
            node_labels = np.load(path).flatten()
            return node_labels
        elif(self.data == 'UKS'):
            path = '/home/embed/Documents/Hyperion_notes/dataset/ukunion/node_label.npy'
            node_labels = np.load(path).flatten()
            return node_labels

        elif self.size == 'large':
            num_nodes = self.num_nodes()
            print("num_nodes: ", num_nodes)
            if self.num_classes == 19:
                #path = '/mnt/smartssd/IGBH/full/processed/paper/node_label_19_extended.npy'
                # path = '/mnt/nvme2n1/IGBH/full/processed/paper/node_label_19.npy'
                path = '/mnt/n3/igb_large/processed/paper/node_label_19.npy'
                # path = '/mnt/nvme2n1/igb_large/processed/paper/node_label_19.npy'
                print("label file: ", path)
                if(self.in_memory):
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes)).copy()
                else:
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 227130858
            else:
                path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_2K_extended.npy'
                
                if(self.in_memory):
                    node_labels = np.load(path)
                else:
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 157675969
        elif self.size == 'full':
            num_nodes = self.num_nodes()
            print("num_nodes: ", num_nodes)
            if self.num_classes == 19:
                #path = '/mnt/smartssd/IGBH/full/processed/paper/node_label_19_extended.npy'
                # path = '/mnt/nvme2n1/IGBH/full/processed/paper/node_label_19.npy'
                path = '/mnt/n3/IGBH/full/processed/paper/node_label_19.npy'
                # path = '/mnt/nvme2n1/igb_large/processed/paper/node_label_19.npy'
                print("label file: ", path)
                if(self.in_memory):
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes)).copy()
                else:
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 227130858
            else:
                path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_2K_extended.npy'
                
                if(self.in_memory):
                    node_labels = np.load(path)
                else:
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 157675969

        else:
            if self.num_classes == 19:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
                print("label file: ", path)
            else:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
            if self.in_memory:
                #print("label test")
                node_labels = np.load(path)
            else:
                node_labels = np.load(path, mmap_mode='r')
        return node_labels

    @property
    def paper_edge(self) -> np.ndarray:
        # 边关系文件
        path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
        if self.data == 'OGB':
            path = osp.join(self.dir, 'edge_index.npy')
        elif self.data == 'UKS':
            path = '/home/embed/Documents/Hyperion_notes/dataset/ukunion/edge_index.npy'
        elif self.data == 'UKL':
            path = '/home/embed/Documents/Hyperion_notes/dataset/uk2014/edge_index.npy'
        elif self.size == 'full':
            path = '/mnt/n3/IGBH/full/processed/paper__cites__paper/edge_index.npy'
        elif self.size == 'large':
            # path = '/mnt/nvme2n1/igb_large/processed/paper__cites__paper/edge_index.npy'
            path = '/mnt/n3/igb_large/processed/paper__cites__paper/edge_index.npy'
            print("edge file: ", path)
        elif self.size == 'small':
            path = '/mnt/n3/igb_datasets/small/processed/paper__cites__paper/edge_index.npy'
            print("edge file: ", path)
        
        if self.in_memory or self.uva_graph: # uva_graph == 1
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')


class IGB260MDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260MDGLDataset')

    def process(self):
        #dataset_start_time = time.time()
        dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, uva_graph=self.args.uva_graph, \
            classes=self.args.num_classes, synthetic=self.args.synthetic, emb_size=self.args.emb_size, data=self.args.data)
        

        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        print(node_edges.shape)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        #dataset_time = time.time() - dataset_start_time
        #print("dataset_time", dataset_time)

        #print("node edge:", node_edges)  # 这个输出会将整个npy文件拿到内存中

        cur_path = osp.join(self.dir, self.args.dataset_size, 'processed')
        
        # self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
        # full_graph
        #print(" test graph")
        if self.args.dataset_size == 'ful':
            edge_row_idx = torch.from_numpy(np.load('/mnt/n3/IGBH/full/processed/paper__cites__paper/edge_index_csc_row_idx.npy'))
            edge_col_idx = torch.from_numpy(np.load('/mnt/n3/IGBH/full/processed/paper__cites__paper/edge_index_csc_col_idx.npy'))
            edge_idx = torch.from_numpy(np.load('/mnt/n3/IGBH/full/processed/paper__cites__paper/edge_index_csc_edge_idx.npy'))
            #path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper__cites__paper/edge_index.npy'
            self.graph = dgl.graph(('csc', (edge_col_idx,edge_row_idx,edge_idx)), num_nodes=node_features.shape[0])
            self.graph  = self.graph.formats('csc')
        elif self.args.dataset_size == 'small':
            print(" test small")
            self.graph = dgl.graph((node_edges[0,:],node_edges[1,:]), num_nodes=node_features.shape[0])
            #self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
        else:
            # 图的结构都是直接全部拿到内存中来
            # 构建图
            self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
        #print("self graph: ", self.graph.formats())
        #self.graph.ndata['edge'] = node_edges


        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        #print("self graph2: ", self.graph.formats())

        
        if self.args.dataset_size != 'ful':
            self.graph = dgl.remove_self_loop(self.graph)
            self.graph = dgl.add_self_loop(self.graph)
        print("self graph3: ", self.graph.formats())
        
        if self.args.dataset_size == 'full':
            #TODO: Put this is a meta.pt file
            if self.args.num_classes == 19:
                n_labeled_idx = 227130858
            else:
                n_labeled_idx = 157675969

            n_nodes = node_features.shape[0]
            n_train = int(n_labeled_idx * 0.6)
            print("n_train", n_train)
            n_val   = int(n_labeled_idx * 0.2)
            print("self graph4: ", self.graph.formats())    
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:n_labeled_idx] = True
            print("self graph5: ", self.graph.formats())
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        else:
            n_nodes = node_features.shape[0]
            print("n_nodes: ", n_nodes)
            '''
            # 假设你已经有 new_seq，长度 N
            new_seq = np.load('new_sequence.npy', mmap_mode='r')
            N = new_seq.shape[0]

            # 1) 构造逆映射 old_id -> new_id
            inv_perm = np.empty(N, dtype=new_seq.dtype)
            inv_perm[new_seq] = np.arange(N, dtype=new_seq.dtype)

            # 2) 原来的比例切分
            n_train = int(N * 0.6)
            n_val   = int(N * 0.2)
            # (测试集大小就是 N - n_train - n_val)

            # 3) 生成新的 mask
            train_mask = torch.zeros(N, dtype=torch.bool)
            val_mask   = torch.zeros(N, dtype=torch.bool)
            test_mask  = torch.zeros(N, dtype=torch.bool)

            # 原老 ID 区间
            old_train_ids = np.arange(0, n_train)
            old_val_ids   = np.arange(n_train, n_train + n_val)
            old_test_ids  = np.arange(n_train + n_val, N)

            # 映射到新 ID 空间
            new_train_ids = inv_perm[old_train_ids]
            new_val_ids   = inv_perm[old_val_ids]
            new_test_ids  = inv_perm[old_test_ids]

            # 填 mask
            train_mask[new_train_ids] = True
            val_mask[new_val_ids]     = True
            test_mask[new_test_ids]   = True

            # 4) 写回图里
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask']   = val_mask
            self.graph.ndata['test_mask']  = test_mask

            # 打印验证
            print(f"Train: {train_mask.sum().item()} nodes, range new_id = {new_train_ids.min()}…{new_train_ids.max()}")
            print(f" Val : {val_mask.sum().item()} nodes, range new_id = {new_val_ids.min()}…{new_val_ids.max()}")
            print(f"Test : {test_mask.sum().item()} nodes, range new_id = {new_test_ids.min()}…{new_test_ids.max()}")

            '''
            n_train = int(n_nodes * 0.6)
            #print("n_train: ", n_train)
            n_val   = int(n_nodes * 0.2)
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:] = True

            train_indices = train_mask.nonzero(as_tuple=True)[0]
            print(f"训练节点总数: {train_indices.numel()}")
            print(f"训练节点索引范围: {train_indices.min().item()} ～ {train_indices.max().item()}")
            
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
            
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)

class OGBDGLDataset(DGLDataset): # papers100M
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    def process(self):

        #dataset_start_time = time.time()
        dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, uva_graph=self.args.uva_graph, \
            classes=self.args.num_classes, synthetic=self.args.synthetic, emb_size=self.args.emb_size, data=self.args.data)

        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        #dataset_time = time.time() - dataset_start_time
        #print("dataset_time", dataset_time)

        print("node edge shape:" , node_edges.shape)
        print("num nodes: ", node_features.shape[0])
        print("num labels: ", node_labels.shape[0])
        self.graph = dgl.graph((node_edges[:,0],node_edges[:,1]), num_nodes=111059956)
        #print("self graph average in-degree", self.graph.in_degrees().sum().item())
        
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        # 先去除自环再添加自环
        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)

        n_nodes = 111059956
        #n_train = int(n_nodes * 0.6)
        #n_val   = int(n_nodes * 0.2)
        print("n_nodes: ", n_nodes)
            
        split_path = osp.join(self.dir,'../split', 'time')
        print("split path: ", split_path)
        train_mask, val_mask, test_mask = ogb_get_idx_split_mask(split_path, n_nodes)
        

        #train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        #val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        #test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
        #train_mask[:n_train] = True
        #val_mask[n_train:n_train + n_val] = True
        #test_mask[n_train + n_val:] = True
            
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)

class UKSDGLDataset(DGLDataset): # UKS
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    def process(self):

        #dataset_start_time = time.time()
        dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, uva_graph=self.args.uva_graph, \
            classes=self.args.num_classes, synthetic=self.args.synthetic, emb_size=self.args.emb_size, data=self.args.data)

        #node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        #dataset_time = time.time() - dataset_start_time
        #print("dataset_time", dataset_time)

        print("node edge shape:" , node_edges.shape)
        #print("num nodes: ", node_features.shape[0])
        print("num labels: ", node_labels.shape[0])
        self.graph = dgl.graph((node_edges[0,:],node_edges[1,:]), num_nodes=133633040)
        #self.graph = dgl.graph((node_edges[:,0],node_edges[:,1]), num_nodes=133633040)
        #print("self graph average in-degree", self.graph.in_degrees().sum().item())
        
        #self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        # 先去除自环再添加自环
        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)

        n_nodes = 133633040

        split_dir = '/home/embed/Documents/Hyperion_notes/dataset/ukunion/'
        #print("Split path:", split_dir)

        # 加载划分 ID
        train_ids = np.fromfile(split_dir + 'trainingset', dtype=np.int32)
        val_ids   = np.fromfile(split_dir + 'validationset', dtype=np.int32)
        test_ids  = np.fromfile(split_dir + 'testingset', dtype=np.int32)

        print(f"Train set size: {len(train_ids):,}")
        print(f"Validation set size: {len(val_ids):,}")
        print(f"Test set size: {len(test_ids):,}")


        # 初始化全 False 的 mask
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(n_nodes, dtype=torch.bool)

        # 将指定位置设置为 True
        train_mask[torch.from_numpy(train_ids)] = True
        val_mask[torch.from_numpy(val_ids)] = True
        test_mask[torch.from_numpy(test_ids)] = True

        # 注册到 DGL 图中
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
            

        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)
    
class UKLDGLDataset(DGLDataset): # UKL
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    def process(self):
        dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, uva_graph=self.args.uva_graph, \
            classes=self.args.num_classes, synthetic=self.args.synthetic, emb_size=self.args.emb_size, data=self.args.data)

        node_edges = torch.from_numpy(dataset.paper_edge)

        print("node edge shape:" , node_edges.shape)
        self.graph = dgl.graph((node_edges[0,:],node_edges[1,:]), num_nodes=787801471)

        # 先去除自环再添加自环
        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)

        #n_nodes = 787801471

class IGBHeteroDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')
    def process(self):

        if self.args.in_memory:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper__cites__paper', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy')))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy')))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy')))

        else:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper__cites__paper', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy'), mmap_mode='r'))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r'))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy'), mmap_mode='r'))

        graph_data = {
            ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
            ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
            ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
            ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1])
        }
        self.graph = dgl.heterograph(graph_data)     
        self.graph.predict = 'paper'

        if self.args.in_memory:
            paper_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_feat.npy')))
            paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_label_19.npy'))).to(torch.long)
        else:
            paper_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_feat.npy'), mmap_mode='r'))
            paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_label_19.npy'), mmap_mode='r')).to(torch.long)  

        self.graph.nodes['paper'].data['feat'] = paper_node_features
        self.graph.num_paper_nodes = paper_node_features.shape[0]
        self.graph.nodes['paper'].data['label'] = paper_node_labels
        if self.args.in_memory:
            author_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author', 'node_feat.npy')))
        else:
            author_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author', 'node_feat.npy'), mmap_mode='r'))
        self.graph.nodes['author'].data['feat'] = author_node_features
        self.graph.num_author_nodes = author_node_features.shape[0]

        if self.args.in_memory:
            institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'institute', 'node_feat.npy')))       
        else:
            institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'institute', 'node_feat.npy'), mmap_mode='r'))
        self.graph.nodes['institute'].data['feat'] = institute_node_features
        self.graph.num_institute_nodes = institute_node_features.shape[0]

        if self.args.in_memory:
            fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'fos', 'node_feat.npy')))       
        else:
            fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'fos', 'node_feat.npy'), mmap_mode='r'))
        self.graph.nodes['fos'].data['feat'] = fos_node_features
        self.graph.num_fos_nodes = fos_node_features.shape[0]
        
        self.graph = dgl.remove_self_loop(self.graph, etype='cites')
        self.graph = dgl.add_self_loop(self.graph, etype='cites')
        
        n_nodes = paper_node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        self.graph.nodes['paper'].data['train_mask'] = train_mask
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask
        

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
class IGBHeteroDGLDatasetMassive(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    def process(self):
        if self.args.uva_graph:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__cites__paper', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy')))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy')))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy')))
        else:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__cites__paper', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy'), mmap_mode='r'))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r'))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy'), mmap_mode='r'))

        if self.args.dataset_size == "full":
            num_paper_nodes = 269346174
            paper_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
            'paper', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes,1024)))
            if self.args.num_classes == 19:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
                'paper', 'node_label_19.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            elif self.args.num_classes == 2983:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
                'paper', 'node_label_2K.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            num_author_nodes = 277220883
            author_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
            'author', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_author_nodes,1024)))

          
        elif self.args.dataset_size == "large":
            num_paper_nodes = 100000000
            paper_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
            'paper', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes,1024)))
            if self.args.num_classes == 19:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
                'paper', 'node_label_19.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            elif self.args.num_classes == 2983:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
                'paper', 'node_label_2K.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            num_author_nodes = 116959896
            author_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
            'author', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_author_nodes,1024)))

        institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'institute', 'node_feat.npy'), mmap_mode='r'))
        fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'fos', 'node_feat.npy'), mmap_mode='r'))
        num_nodes_dict = {'paper': num_paper_nodes, 'author': num_author_nodes, 'institute': len(institute_node_features), 'fos': len(fos_node_features)}
        print("Setting the graph structure")
        graph_data = {
            ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
            ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
            ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
            ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1])
        }
        #graph_data = torch.load("/mnt/nvme22/IGBH_csc.pth")
        print("dgl.heterograph init starting")
        self.graph = dgl.heterograph(graph_data, num_nodes_dict) 
        self.graph = self.graph.formats('csc')
        print(self.graph.formats())
        print("dgl.heterograph init successful")
        self.graph.predict = 'paper'
        #self.graph = dgl.remove_self_loop(self.graph, etype='cites')
        #self.graph = dgl.add_self_loop(self.graph, etype='cites')
        self.graph.nodes['paper'].data['feat'] = paper_node_features
        self.graph.num_paper_nodes = paper_node_features.shape[0]
        self.graph.nodes['paper'].data['label'] = paper_node_labels
        self.graph.nodes['author'].data['feat'] = author_node_features
        self.graph.num_author_nodes = author_node_features.shape[0]

        self.graph.nodes['institute'].data['feat'] = institute_node_features
        self.graph.num_institute_nodes = institute_node_features.shape[0]

        self.graph.nodes['fos'].data['feat'] = fos_node_features
        self.graph.num_fos_nodes = fos_node_features.shape[0]
        
        n_nodes = paper_node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        self.graph.nodes['paper'].data['train_mask'] = train_mask
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
class OGBHeteroDGLDatasetMassive(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    def process(self):
        if(self.args.uva_graph or self.args.in_memory):
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir,  'processed','author___affiliated_with___institution', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, 'processed', 'author___writes___paper', 'edge_index.npy')))
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir,  'processed',          'paper___cites___paper', 'edge_index.npy')))


        else:
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir,  'processed','author___affiliated_with___institution', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, 'processed', 'author___writes___paper', 'edge_index.npy'), mmap_mode='r'))
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir,  'processed',          'paper___cites___paper', 'edge_index.npy'), mmap_mode='r'))

        num_paper_nodes = 121751666


#        print("a u e: ", affiliation_author_edges[0])
#        print("affiliation_author_edges: ", len(affiliation_author_edges))
#        print("autho paper len: ", len(author_paper_edges))
#        print("paper_paper_edges: ", len(paper_paper_edges))

        print("paper node feature load")
        #paper_node_features = torch.from_numpy(np.load(osp.join(self.dir, 'processed', 'paper', 'node_feat.npy'), mmap_mode='r')).to(torch.float32)
        
        #paper_node_features = torch.from_numpy(np.memmap(osp.join(self.dir,   'processed',  'paper', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes,768)))
        paper_node_features = torch.from_numpy(np.memmap("/mnt/nvme22/MAG_node_feat_mmap.npy",  dtype='float32', mode='r',  shape=(num_paper_nodes,768)))
        print("node feature tensor dim: ", paper_node_features.shape)
#        paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir,  'processed', 'paper', 'node_label.npy'), dtype='long', mode='r',  shape=(num_paper_nodes))).to(torch.long)
        paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, 'processed', 'paper', 'node_label.npy'), mmap_mode='r')).to(torch.long)
        paper_node_labels[paper_node_labels<0]=0
        num_author_nodes = 122383112
        num_institute = 25721
        
        print("min label: ", torch.min(paper_node_labels))
        print("max label: ", torch.max(paper_node_labels))
        #num_author_nodes = 0
        #num_institute = 0
        #num_fos = 0


        num_nodes_dict = {'paper': num_paper_nodes, 'author': num_author_nodes, 'institute': num_institute  }
        #graph_data = {
       #     ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
       #     ('author', 'writes', 'paper'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
       #     ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1])
       # }

       # graph_data = {
       #     ('paper', 'cites', 'paper'): (paper_paper_edges[0,:], paper_paper_edges[1,:]),
       #     ('author', 'writes', 'paper'): (author_paper_edges[0,:], author_paper_edges[1,:]),
       #     ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[0,:], affiliation_author_edges[1,:])
       # }

        graph_data = torch.load("/mnt/nvme22/OGB_csc.pth")
        print("dgl.heterograph init starting")
        self.graph = dgl.heterograph(graph_data, num_nodes_dict)

        self.graph =  self.graph.formats('csc')

        print("dgl.heterograph init successful")
        self.graph.predict = 'paper'
#        self.graph = dgl.remove_self_loop(self.graph, etype='cites')
#        self.graph = dgl.add_self_loop(self.graph, etype='cites')
        
        self.graph.nodes['paper'].data['feat'] = paper_node_features
        self.graph.num_paper_nodes = paper_node_features.shape[0]
        self.graph.nodes['paper'].data['label'] = paper_node_labels
        
        n_nodes = paper_node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True

        self.graph.nodes['paper'].data['train_mask'] = train_mask
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/mnt/nvme16/IGB260M_part_2', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='full',
        choices=['experimental', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=2983, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    args = parser.parse_args()
    start = time.time()

    dataset =  IGBHeteroDGLDatasetMassive(args)
    g = dataset[0]
    print(g)
    homo_g = dgl.to_homogeneous(g)
    print("Time taken: ", time.time() - start)
