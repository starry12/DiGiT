from ogb.nodeproppred import NodePropPredDataset
import numpy as np

dataset = NodePropPredDataset(name = 'ogbn-papers100M',root='/home/embed/OGB')


graph, label = dataset[0] # graph: library-agnostic graph object
np.save('/home/embed/OGB/ogbn_papers100M/raw'+'node_label.npy',label)

features = graph['node_feat'].astype(np.float32)
np.save('/home/embed/OGB/ogbn_papers100M/raw'+'node_feat.npy',features)

edge_index = graph['edge_index'].T
np.save('/home/embed/OGB/ogbn_papers100M/raw'+'edge_index.npy',edge_index)




