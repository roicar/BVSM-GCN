"""
Here we can infer GCN with DGL+Pytorch
We keep the train module in the front

"""

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import timeit
import numpy as np

#Notice:You need to change parameter of infeat_nums after changing dataset
#dataset = dgl.data.TUDataset("AIDS")
#dataset = dgl.data.TUDataset("BZR")
#dataset = dgl.data.TUDataset("COX2")
#dataset = dgl.data.TUDataset("DHFR")
dataset = dgl.data.TUDataset("COX2")

######################################################################
# The dataset is a set of graphs, each with node features and a single
# label. One can see the node feature dimensionality and the number of
# possible graph categories of ``GINDataset`` objects in ``dim_nfeats``
# and ``gclasses`` attributes.
#

print("Max number of node:", dataset.max_num_node)
print("Number of graph categories:", dataset.num_classes)


from dgl.dataloading import GraphDataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler

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
    dataset, sampler=test_sampler, batch_size=1, drop_last=False
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

# Here we choose to infer the whole dataset
# We can also choose the batch_size here
from torch.utils.data import Subset
num_examples = len(dataset)
test_sampler = Subset(dataset, torch.arange(num_examples))
dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=5, drop_last=False
)

'''
#-----------------------------------DenseGCNConv

DMM_time = 0
time1 = 0
time2 = 0
time3 = 0
time4 = 0
class DenseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(DenseGCNConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)


    def forward(self, adj, features):
        #print("OK3")
        adj = torch.from_numpy(adj)
        dense_adj = adj.to_dense()
        #print("dense_adj ",dense_adj)
        start = time.time()
        dense_feat = torch.matmul(dense_adj,features)
        end = time.time()
        global DMM_time
        DMM_time += (end-start)
        
        return self.linear(dense_feat)




#from dgl.nn import GraphConv


class DenseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(DenseGCN, self).__init__()
        self.conv1 = DenseGCNConv(in_feats,h_feats)
        self.conv2 = DenseGCNConv(h_feats,num_classes)
        #print("OK5")
    
    def normalize_adjacency(self, g):
        '''
        adj = g.adjacency_matrix()
        adj = torch.eye(g.number_of_nodes()).to(adj.device) + adj  # 自连接
        deg = adj.sum(1)
        d_inv_sqrt = torch.pow(deg, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        adj = adj * d_inv_sqrt.unsqueeze(1)
        adj_normalized = adj * d_inv_sqrt.unsqueeze(0)
        '''
        # 将PyTorch张量转换为NumPy数组
        adj = g.adjacency_matrix()
        adj = adj.to_dense()
        adj_np = adj.numpy()

        # 计算度数并转换为逆平方根
        deg = np.sum(adj_np, axis=1)
        d_inv_sqrt = np.power(deg, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0

        # 使用 np.multiply 进行归一化
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        start = time.time()
        adj_normalized = np.multiply(np.multiply(d_mat_inv_sqrt, adj_np), d_mat_inv_sqrt)
        end = time.time()
        global time1
        time1 += (end-start)
        # 如果需要，将结果转换回PyTorch张量
        adj_normalized_torch = torch.from_numpy(adj_normalized)
        return adj_normalized
    
        
    def forward(self, g, features):
        #print("g: ",g)
        #start = time.time()
        adj_normalized = self.normalize_adjacency(g)
        #end = time.time()
        #global time1
        #time1 += (end-start)
                
        start = time.time()
        x = F.relu(self.conv1(adj_normalized, features))
        end = time.time()
        global time2
        time2 += (end-start)
        
        start = time.time()
        x = self.conv2(adj_normalized, x)
        end = time.time()
        global time3
        time3 += (end-start)
        
        start = time.time()
        g.ndata['h'] = x
        meannode = dgl.mean_nodes(g, 'h')
        end = time.time()
        global time4
        time4 += (end-start)
        
        return meannode
        '''
        # 将邻接矩阵转换为稠密格式并添加自连接
        #print("OK6")
        adj = g.adjacency_matrix().to_dense() + torch.eye(g.number_of_nodes())
        # print("adj: ",adj)
        # 计算度矩阵的逆平方根
        deg_inv_sqrt = adj.sum(dim=1).pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)
        # 计算标准化邻接矩阵
        adj_normalized = torch.matmul(torch.matmul(deg_inv_sqrt, adj), deg_inv_sqrt)
        
        x = F.relu(self.conv1(adj_normalized, features))
        x = self.conv2(adj_normalized, x)        
        g.ndata['h'] = x
        return dgl.mean_nodes(g, 'h')
        '''



# Create the model with given dimensions
#print("OK7")
model = DenseGCN(3, 16, dataset.num_classes)
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
state_dict = torch.load('COX2_model.pth')
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
for batched_graph, labels in test_dataloader:
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
    
print("total DMM_time: ",DMM_time)
print("num_correct: ",num_correct)
print("num_tests: ",num_tests)
print("Test accuracy:", num_correct / num_tests)
print("Infer time:", elapsed_time)
print("Normalization time:", time1)
print("conv1 time:", time2)
print("conv2 time:", time3)
print("mean_node time:", time4)
