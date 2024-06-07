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
dataset = dgl.data.TUDataset("COX2")


print("Max number of node:", dataset.max_num_node)
print("Number of graph categories:", dataset.num_classes)
max_dimension = dataset.max_num_node
num_features = 3


# 定义填充函数
def pad_graphs(graphs, num_features):
    print("graphs:",graphs)
    max_nodes = max(graph.num_nodes() for graph in graphs)
    print("max_nodes:",max_nodes)
    padded_adjs = []
    padded_features = []
    for graph in graphs:
        # 邻接矩阵
        adj = graph.adjacency_matrix().to_dense()
        adj_padded = F.pad(adj, (0, max_nodes - adj.shape[0], 0, max_nodes - adj.shape[1]))
        
        # 特征矩阵
        
        feat = graph.ndata["node_attr"]
        #print("feat:",feat)
        feat_padded = F.pad(feat, (0, 0, 0, max_nodes - feat.shape[0]))
        
        padded_adjs.append(adj_padded)
        padded_features.append(feat_padded)
        
    return torch.stack(padded_adjs), torch.stack(padded_features)

# 创建数据加载器
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    padded_adjs, padded_features = pad_graphs(graphs, num_features)
    return padded_adjs, padded_features, torch.tensor(labels)


# 数据加载器
data_loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate)


#-----------------------------------DenseGCNConv
class DenseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(DenseGCNConv, self).__init__()
        self.linear = nn.Linear(in_feats,out_feats)

    def forward(self, adj, features):
        #print("OK3")
        dense_adj = adj.to_dense()
        #print("shape: ",dense_adj.shape())
        dense_feat = torch.matmul(dense_adj.to(torch.float),features.to(torch.float))
        #print("OK4")
        #support = torch.mm(input_features, self.weight)
        #output = torch.mm(adjacency_matrix, support)
        return self.linear(dense_feat)




#from dgl.nn import GraphConv


class DenseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(DenseGCN, self).__init__()
        self.conv1 = DenseGCNConv(in_feats, h_feats)
        self.conv2 = DenseGCNConv(h_feats, num_classes)
        
    
    def forward(self, adjs, features):
        max_nodes = adjs.shape[1]
        '''
        x = F.relu(self.conv1(adjs,features))
        x = self.conv2(adjs, x)
        x = x.sum(dim=1)/max_nodes
        '''
        # 从邻接矩阵的大小推断出每个批次的最大节点数
        max_nodes = adjs.shape[1]  # 假设邻接矩阵是方形的，所以取第二维的大小
        
        # 添加自连接
        I = torch.eye(max_nodes).to(adjs.device)  # 创建单位矩阵，并确保其在相同的设备上
        adjs_with_self_loops = adjs + I
        
        # 计算度矩阵的逆平方根
        degree_matrix_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(adjs_with_self_loops.sum(dim=1) + 1e-6))  # 避免除以0
        
        # 归一化邻接矩阵
        adjs_normalized = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, adjs_with_self_loops), degree_matrix_inv_sqrt)
        
        x = F.relu(self.conv1(adjs_normalized, features))
        x = self.conv2(adjs_normalized, x)
        
        # 使用动态计算的max_nodes进行归一化
        x = x.sum(dim=1) / max_nodes
        
        return x



# Create the model with given dimensions
# print("OK7")
model = DenseGCN(3, 16, dataset.num_classes)
# print("OK8")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


##########################################################################
# Training Loop
# Skip it when you need inference
'''
loss_func = nn.CrossEntropyLoss()
for epoch in range(75):
    #for batched_graph, labels in data_loader:
    for adjs, features, labels in data_loader:
        # 模型前向传播和训练...
        pred = model(adjs,features)
        loss = loss_func(pred,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch (epoch):loss {loss.item()}')

#save parameters
torch.save(model.state_dict(),'COX2_DGL_M.pth')
'''

# pass the parameters

state_dict = torch.load('COX2_model.pth')
'''
new_state_dict = {}

# Update the keys
for key in state_dict:
    new_key = key.replace("conv1.", "conv1.linear.").replace("conv2.", "conv2.linear.")
    new_state_dict[new_key] = state_dict[key]

model.load_state_dict(new_state_dict)
'''
model.eval()

num_correct = 0
num_tests = 0
elapsed_time = 0
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
print("Max number of node:", dataset.max_num_node)
print("Number of graph categories:", dataset.num_classes)
print("num_correct: ",num_correct)
print("num_tests: ",num_tests)
print("Test accuracy:", num_correct / num_tests)
print("Infer time:", elapsed_time)

