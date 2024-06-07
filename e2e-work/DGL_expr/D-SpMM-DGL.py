"""
Here we can infer GCN with DGL+Pytorch
We keep the train module in the front

"""

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import timeit


#Notice:You need to change parameter of infeat_nums after changing dataset
#dataset = dgl.data.TUDataset("AIDS")
#dataset = dgl.data.TUDataset("BZR")
#dataset = dgl.data.TUDataset("COX2")
#dataset = dgl.data.TUDataset("DHFR")
dataset = dgl.data.TUDataset("AIDS")
# 1. 读取COX2数据集

# 2. 获取图的节点数并存储（图的索引，节点数）
graph_sizes = [(i, g[0].num_nodes()) for i, g in enumerate(dataset)]

# 3. 根据节点数排序
sorted_indices = [i for i, _ in sorted(graph_sizes, key=lambda x: x[1])]

# 4. 创建一个新的数据集列表，按照排序后的索引
sorted_dataset = [dataset[i] for i in sorted_indices]



# 现在data_loader中的数据将按图的节点数从小到大排列
######################################################################
# The dataset is a set of graphs, each with node features and a single
# label. One can see the node feature dimensionality and the number of
# possible graph categories of ``GINDataset`` objects in ``dim_nfeats``
# and ``gclasses`` attributes.
#

print("Max number of node:", dataset.max_num_node)
num_features = 4

from dgl.dataloading import GraphDataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler


'''
num_examples = len(dataset)
#num_train = int(num_examples * 0.01)
num_test = int(num_examples)

#train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SequentialSampler(torch.arange(num_test))

#train_dataloader = GraphDataLoader(
#    dataset, sampler=train_sampler, batch_size=5, drop_last=False
#)

# Here we compare with BVSM-M, batch_size=num_test
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=32, drop_last=False
)




it = iter(test_dataloader)
batch = next(it)
print(batch)
batched_graph, labels = batch
print(
    "Number of nodes for each graph element in the batch:",
    batched_graph.batch_num_nodes(),
)
print(
    "Number of edges for each graph element in the batch:",
    batched_graph.batch_num_edges(),
)

# Recover the original graph elements from the minibatch
graphs = dgl.unbatch(batched_graph)
print("The original graphs in the minibatch:")
#print(graphs)


'''
# 定义填充函数
def preprocess_graphs(graphs, num_features):
    dense_adjs = []
    features = []
    for graph in graphs:

        # 邻接矩阵
        adj = graph.adjacency_matrix().to_dense()
        adj = torch.eye(graph.number_of_nodes()).to(adj.device) + adj
        # 特征矩阵
        feat = graph.ndata["node_attr"]
        
        dense_adjs.append(adj)
        features.append(feat)
        
    return torch.stack(dense_adjs), torch.stack(features)

# 创建数据加载器
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    dense_adjs, features = preprocess_graphs(graphs,num_features)
    return dense_adjs, features, torch.tensor(labels)


# 数据加载器
data_loader = DataLoader(sorted_dataset, batch_size=1, shuffle=False, collate_fn=collate)



#-----------------------------------DenseGCNConv

DMM_time = 0
DAD_time = 0
OtherNormalization_time = 0
time1 = 0
time2 = 0
time3 = 0
time4 = 0
linear_time = 0
class DenseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(DenseGCNConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)


    def forward(self, adj, features):
        #print("OK3")
        
        #dense_adj = adj.to_dense()
        #print("dense_adj ",dense_adj)
        adj = adj.to(torch.float)
        features = features.to(torch.float)
        start = time.time()
        dense_feat = torch.matmul(adj,features)
        end = time.time()
        global DMM_time
        DMM_time += (end-start)
        print("This graph Transformed denseMM time is :", (end - start))

        time_5 = time.time()
        rel = self.linear(dense_feat)
        time_6 = time.time()
        global linear_time
        linear_time += (time_6 - time_5)
        return rel




#from dgl.nn import GraphConv


class DenseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(DenseGCN, self).__init__()
        self.conv1 = DenseGCNConv(in_feats,h_feats)
        self.conv2 = DenseGCNConv(h_feats,num_classes)
        #print("OK5")
    
    def normalize_adjacency(self, adj):

        time_1 = time.time()
        #max_nodes = adj.shape[1]
        #adj = torch.eye(max_nodes).to(adjs.device) + adjs  # 自连接
        deg = adj.sum(1)
        d_inv_sqrt = torch.pow(deg, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        time_2 = time.time()
        global OtherNormalization_time
        OtherNormalization_time += (time_2 - time_1)
        
        time_3 = time.time()
        adj = adj * d_inv_sqrt.unsqueeze(1)
        adj_normalized = adj * d_inv_sqrt.unsqueeze(0)
        time_4 = time.time()
        global DAD_time
        DAD_time += (time_4 - time_3)
        return adj_normalized
    
        
    def forward(self, adjs, features):
        
        start1 = time.time()
        #adjs = g.adjacency_matrix().to_dense()
        adj_normalized = self.normalize_adjacency(adjs)
        end1 = time.time()
        global time1
        time1 += (end1-start1)
                
        start2 = time.time()
        x = self.conv1(adj_normalized, features)
        end2 = time.time()
        global time2
        time2 += (end2-start2)

        x = F.relu(x)
        start3 = time.time()
        x = self.conv2(adj_normalized, x)
        end3 = time.time()
        global time3
        time3 += (end3 - start3)
        
        start4 = time.time()
        '''
        g.ndata['h'] = x
        meannode = dgl.mean_nodes(g, 'h')
        '''
        max_nodes = adjs.shape[1]
        x = x.sum(dim=1) / max_nodes
        end4 = time.time()
        global time4
        time4 += (end4-start4)
        
        return x
       
        



# Create the model with given dimensions
#print("OK7")
model = DenseGCN(4, 16, dataset.num_classes)
#print("OK8")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


##########################################################################
# Training Loop
# Skip it when you only need inference
'''

for epoch in range(75):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata["node_attr"].float())
        #print("batched_graph: ",batched_graph)
        #print("batched_graph.ndata[attr] ", batched_graph.ndata["node_attr"].float())
        #print("pred: ",pred)
        #print("label: ",labels)
        loss = F.cross_entropy(pred, labels.squeeze(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch (epoch):loss {loss.item()}')

#save parameters
torch.save(model.state_dict(),'BZR_model_DSpMM.pth')
# pass the parameters
'''

#state_dict = torch.load('COX2_model_Dense.pth')
state_dict = torch.load('../model_parameters/AIDS_model.pth')
#print("state_dict: ",state_dict)

new_state_dict = {}
state_dict["conv1.weight"] = state_dict["conv1.weight"].t()
state_dict["conv2.weight"] = state_dict["conv2.weight"].t()

# Update the keys
for key in state_dict:
    new_key = key.replace("conv1.", "conv1.linear.").replace("conv2.", "conv2.linear.")
    new_state_dict[new_key] = state_dict[key]

model.load_state_dict(new_state_dict)

model.eval()

num_correct = 0
num_tests = 0
elapsed_time = 0
test_num = 0
'''
for batched_graph, labels in data_loader:
    print("test_num: ",test_num)
    test_num += 128
    #print("batched_graph: ",batched_graph)
    length = len(labels)
    print("-----------------------------------------------------")
    #print("labels: ",labels)
    #print("labels.squeeze(): ",labels.squeeze())
    labels = labels.squeeze()
    start = time.time()
    pred = model(batched_graph, batched_graph.ndata["node_attr"].float())
    end = time.time()
    elapsed_time = elapsed_time + (end - start)
    print("pred: ",pred)
    print("pred.argmax(1): ",pred.argmax(1))
    print("pred.argmax(1).squeeze(): ",pred.argmax(1).squeeze())
    print("(pred.argmax(1)==labels).sum: ",(pred.argmax(1) == labels).sum())
    print("(pred.argmax(1)==labels).sum.item(): ",(pred.argmax(1) == labels).sum().item())
    print("num_correct_before:",num_correct)
    num_correct += (pred.argmax(1) == labels).sum().item()
    print("num_correct_after:",num_correct)
    num_tests += length
'''
for adjs, features, labels in data_loader:
    #print("batched_graph: ",batched_graph)
    length = len(labels)
    print("-----------------------------------------------------")
    print("labels: ",labels)
    print("labels.squeeze(): ",labels.squeeze())
    labels = labels.squeeze()
    start = time.time()
    pred = model(adjs,features)
    end = time.time()
    elapsed_time = elapsed_time + (end - start)
    print("pred: ",pred)
    print("pred.argmax(1): ",pred.argmax(1))
    print("pred.argmax(1).squeeze(): ",pred.argmax(1).squeeze())
    print("(pred.argmax(1)==labels).sum: ",(pred.argmax(1) == labels).sum())
    print("(pred.argmax(1)==labels).sum.item(): ",(pred.argmax(1) == labels).sum().item())
    print("num_correct_before:",num_correct)
    num_correct += (pred.argmax(1) == labels).sum().item()
    print("num_correct_after:",num_correct)
    num_tests += length  

print("OtherNormalization_time:", OtherNormalization_time)
print("DAD_time:", DAD_time)
#print("Normalization time:", time1)
print("MM time:", time2 + time3)
print("Key_time: ",DMM_time)
print("mean_node time:", time4)
print("linear_time: ", linear_time)

print("num_correct: ",num_correct)
print("num_tests: ",num_tests)
print("Test accuracy:", num_correct / num_tests)
print("Infer time:", elapsed_time)

