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
max_nodes = dataset.max_num_node
num_features = 3


# 定义填充函数
def pad_graphs(graphs, max_nodes, num_features):
    padded_adjs = []
    padded_features = []
    for graph in graphs:
        # Calculate the nearest multiple of 8 for max_nodes
        if max_nodes % 8 != 0:  # If max_nodes is not already a multiple of 8
            padded_max_nodes = ((max_nodes // 8) + 1) * 8
        else:
            padded_max_nodes = max_nodes
        
        # 邻接矩阵
        adj = graph.adjacency_matrix().to_dense()
        # Adjust padding to the new padded_max_nodes
        adj_padded = F.pad(adj, (0, padded_max_nodes - adj.shape[0], 0, padded_max_nodes - adj.shape[1]))
        
        # 特征矩阵
        feat = graph.ndata["node_attr"]
        # Adjust padding to the new padded_max_nodes
        feat_padded = F.pad(feat, (0, 0, 0, padded_max_nodes - feat.shape[0]))
        
        padded_adjs.append(adj_padded)
        padded_features.append(feat_padded)
        
    return torch.stack(padded_adjs), torch.stack(padded_features)

# 创建数据加载器
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    padded_adjs, padded_features = pad_graphs(graphs, max_nodes, num_features)
    return padded_adjs, padded_features, torch.tensor(labels)


# 数据加载器
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate)

DAD_time = 0
KeyMM_time = 0
MM_time = 0
mean_time = 0
OtherNormalization_time = 0
#-----------------------------------DenseGCNConv
class DenseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(DenseGCNConv, self).__init__()
        self.linear = nn.Linear(in_feats,out_feats)

    def forward(self, adj, features):
        #print("OK3")
        #dense_adj = adj.to_dense()
        #dense_adj = adj
        #adj has been transformed into dense format
        #print("shape: ",dense_adj.shape())
        time_1 = time.time()
        dense_feat = torch.matmul(adj.to(torch.float),features.to(torch.float))
        time_2 = time.time()
        global KeyMM_time 
        KeyMM_time += (time_2 - time_1)
        #print("OK4")
        #support = torch.mm(input_features, self.weight)
        #output = torch.mm(adjacency_matrix, support)
        return self.linear(dense_feat)




#from dgl.nn import GraphConv



class DenseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, max_nodes):
        super(DenseGCN, self).__init__()
        self.conv1 = DenseGCNConv(in_feats, h_feats)
        self.conv2 = DenseGCNConv(h_feats, num_classes)
        self.max_nodes = max_nodes
    
    def forward(self, adjs, features):
        
        time_3 = time.time()
        #max_nodes = adjs.shape[1]
        # 从邻接矩阵的大小推断出每个批次的最大节点数
        max_nodes = adjs.shape[1]  # 假设邻接矩阵是方形的，所以取第二维的大小
        
        # 添加自连接
        I = torch.eye(max_nodes).to(adjs.device)  # 创建单位矩阵，并确保其在相同的设备上
        adjs_with_self_loops = adjs + I
        
        # 计算度矩阵的逆平方根
        degree_matrix_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(adjs_with_self_loops.sum(dim=1) + 1e-6))  # 避免除以0
        time_4 = time.time()
        global OtherNormalization_time
        OtherNormalization_time += (time_4 - time_3)
        
        time_9 = time.time()
        # 归一化邻接矩阵
        #adjs_normalized = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, adjs_with_self_loops), degree_matrix_inv_sqrt)
        adjs = adjs_with_self_loops * degree_matrix_inv_sqrt.unsqueeze(1)
        adjs_normalized = adjs * degree_matrix_inv_sqrt.unsqueeze(0)
        time_10 = time.time()
        global DAD_time
        DAD_time += (time_10 - time_9)
        
        time_5 = time.time()
        x = F.relu(self.conv1(adjs_normalized, features))
        x = self.conv2(adjs_normalized, x)
        time_6 = time.time()
        global MM_time
        MM_time += (time_6 - time_5)
        
        time_7 = time.time()
        # 使用动态计算的max_nodes进行归一化
        # x = x.sum(dim=1) / max_nodes
        x = x.mean(dim=1)
        time_8 = time.time()
        global mean_time
        mean_time += (time_8 - time_7)
        
        return x



# Create the model with given dimensions
# print("OK7")
model = DenseGCN(3, 16, dataset.num_classes, max_nodes)
# print("OK8")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


##########################################################################
# Training Loop
# Skip it when you need inference


# pass the parameters

state_dict = torch.load('COX2_model.pth')

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
    
print("DAD_time: ",DAD_time)
print("OtherNormalization_time: ",OtherNormalization_time)
print("MM_time: ",MM_time)
print("KeyMM_time: ",KeyMM_time)
print("mean_time: ",mean_time)

print("num_correct: ",num_correct)
print("num_tests: ",num_tests)
print("Test accuracy:", num_correct / num_tests)
print("Infer time:", elapsed_time)

