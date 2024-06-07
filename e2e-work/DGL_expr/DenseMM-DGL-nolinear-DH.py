'''
Created on March 15, 2024
@author: Dai

TODO: We tryed DAD and dense MM in GCN and we created the Sparse version for comparison.Here we want to see the performance lift of only dense.
Thus, we keep the normalization as same as DGL.py.
'''

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
from torch.utils.data import DataLoader,SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import time
import timeit
import numpy as np

#dataset = dgl.data.TUDataset("AIDS")
#dataset = dgl.data.TUDataset("BZR")
#dataset = dgl.data.TUDataset("COX2")
#dataset = dgl.data.TUDataset("DHFR")
dataset = dgl.data.TUDataset("AIDS")
# 1. 读取数据集

# 2. 获取图的节点数并存储（图的索引，节点数）
graph_sizes = [(i, g[0].num_nodes()) for i, g in enumerate(dataset)]

# 3. 根据节点数排序
sorted_indices = [i for i, _ in sorted(graph_sizes, key=lambda x: x[1])]

# 4. 创建一个新的数据集列表，按照排序后的索引
sorted_dataset = [dataset[i] for i in sorted_indices]

subset = int(len(sorted_dataset))
inference_subset = sorted_dataset[:subset]

# 现在data_loader中的数据将按图的节点数从小到大排列


print("Max number of node:", dataset.max_num_node)
num_features = 3

from dgl.dataloading import GraphDataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler



# PreProcess Stage
def preprocess_graphs(graphs):
    processed_graphs = []
    for g in graphs:
        #print("g:",g)
        t = g[1]
        g = g[0]
        # add selfloop
        #print("g:",g)
        g = dgl.add_self_loop(g)

        # normalization
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs,-0.5)
        #norm = norm.to(g.device).unsqueeze(1)
        #print(g)
        g.ndata['norm'] = norm
        processed_graphs.append((g,t))
    return processed_graphs



dataset = preprocess_graphs(inference_subset)
from torch.utils.data.sampler import SequentialSampler
num_examples = len(dataset)
#num_train = int(num_examples * 0.01)
num_test = int(num_examples )

# test for one graph
# num_test = 1

#train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SequentialSampler(torch.arange(num_test))

#train_dataloader = GraphDataLoader(
#    dataset, sampler=train_sampler, batch_size=5, drop_last=False
#)

# Here we compare with BVSM-M, batch_size=num_test
start = time.time()

test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=1, drop_last=False
)
end = time.time()
print("The Dataload time is : ", (end - start))

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


#-----------------------------------DenseGCNConv

KeyMM_1_time = 0
KeyMM_2_time = 0
linear_1_time = 0
linear_2_time = 0
bias_transform_1_time = 0
bias_transform_2_time = 0
bias_1_time = 0
bias_2_time = 0
meanNode_time = 0
torch_transform_1_time = 0
torch_transform_2_time = 0
Htonumpy_1_time = 0
Htonumpy_2_time = 0
Wtonumpy_1_time = 0
Wtonumpy_2_time = 0
adj_todense = 0
get_adj_from_graph = 0
adj_tonumpy_1_time = 0
adj_tonumpy_2_time = 0

get_norm = 0
norm_transform = 0
DAD_time = 0


Normalize_1_1_time = 0
Normalize_1_2_time = 0
Normalize_2_1_time = 0
Normalize_2_2_time = 0
conv1 = 0
relu = 0
conv2 = 0

class DenseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(DenseGCNConv, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_feats, out_feats))
        self.bias = nn.Parameter(torch.randn(out_feats))
        self.infeats = in_feats
        self.out_feats = out_feats
        # 移除原有的self.linear

    def forward(self, adj, features, norm):
        # (DH)
        start = time.time()
        features = features * norm
        end = time.time()
        if self.out_feats == 16:
            global Normalize_1_1_time
            Normalize_1_1_time += (end - start)
        else:
            global Normalize_2_1_time
            Normalize_2_1_time += (end - start)

        # A(Pytorch to numpy)
        start = time.time()
        adj = adj.to(torch.float).detach().numpy()  # 确保转换为numpy数组
        end = time.time()
        if self.out_feats == 16:
            global adj_tonumpy_1_time
            adj_tonumpy_1_time += (end - start)
        else:
            global adj_tonumpy_2_time
            adj_tonumpy_2_time += (end - start)

        # DH(Pytorch to numpy)
        start = time.time()
        features = features.to(torch.float).detach().numpy()  # 确保转换为numpy数组
        end = time.time()
        if self.out_feats == 16:
            global Htonumpy_1_time
            Htonumpy_1_time += (end - start)
        else:
            global Htonumpy_2_time
            Htonumpy_2_time += (end - start)

        # W(Pytorch to numpy)
        start = time.time()
        weight = self.weight.detach().numpy()
        end = time.time()
        if self.out_feats == 16:
            global Wtonumpy_1_time
            Wtonumpy_1_time += (end - start)
        else:
            global Wtonumpy_2_time
            Wtonumpy_2_time += (end - start)


        if self.infeats < self.out_feats:
            # first compute A(DH)
            start = time.time()
            dense_feat = np.dot(adj, features)  # adj和特征的乘积
            end = time.time()
            if self.out_feats == 16:
                global KeyMM_1_time
                KeyMM_1_time += (end - start)
            else:
                global KeyMM_2_time
                KeyMM_2_time += (end - start)

            # then compute (A(DH))W
            start = time.time()
            dense_feat = np.dot(dense_feat, weight)
            end = time.time()
            if self.out_feats == 16:
                global linear_1_time
                linear_1_time += (end - start)
            else:
                global linear_2_time
                linear_2_time += (end - start)
        else:
            # (DH)W
            start = time.time()
            dense_feat = np.dot(features, weight)
            end = time.time()
            if self.out_feats == 16:
                linear_1_time += (end - start)
            else:
                linear_2_time += (end - start)

            # A((DH)W)
            start = time.time()
            dense_feat = np.dot(adj, dense_feat)  # adj和特征的乘积
            end = time.time()
            if self.out_feats == 16:
                KeyMM_1_time += (end - start)
            else:
                KeyMM_2_time += (end - start)

        # A((DH)W) (From numpy to Torch)
        start = time.time()
        dense_feat = torch.from_numpy(dense_feat)
        end = time.time()
        if self.out_feats == 16:
            global torch_transform_1_time
            torch_transform_1_time += (end - start)
        else:
            global torch_transform_2_time
            torch_transform_2_time += (end - start)

        # D(A((DH)W))
        start = time.time()
        dense_feat = dense_feat * norm
        end = time.time()
        if self.out_feats == 16:
            global Normalize_1_2_time
            Normalize_1_2_time += (end - start)
        else:
            global Normalize_2_2_time
            Normalize_2_2_time += (end - start)

        # D(A((DH)W)) + bias
        if self.bias is not None:
            # start = time.time()
            bias = self.bias.detach()
            # end = time.time()
            # if self.out_feats == 16:
            #     global bias_transform_1_time
            #     bias_transform_1_time += (end - start)
            # else:
            #     global bias_transform_2_time
            #     bias_transform_2_time += (end - start)

            start = time.time()
            dense_feat += bias
            end = time.time()
            if self.out_feats == 16:
                global bias_1_time
                bias_1_time += (end - start)
            else:
                global bias_2_time
                bias_2_time += (end - start)

        # start = time.time()
        # dense_feat = torch.from_numpy(dense_feat)
        # end = time.time()
        # if self.out_feats == 16:
        #     global torch_transform_1_time
        #     torch_transform_1_time += (end - start)
        # else:
        #     global torch_transform_2_time
        #     torch_transform_2_time += (end - start)
        return dense_feat

#from dgl.nn import GraphConv
class DenseGCN(nn.Module):
    '''
    Dense GCN model

    We first compute DH in forward part.
    H' = sigma(D[A(DH)W] + b)
    '''
    def __init__(self, in_feats, h_feats, num_classes):
        super(DenseGCN, self).__init__()
        self.conv1 = DenseGCNConv(in_feats,h_feats)
        self.conv2 = DenseGCNConv(h_feats,num_classes)

    def forward(self, g, features):
        start = time.time()
        adj = g.adjacency_matrix()
        end = time.time()
        global get_adj_from_graph
        get_adj_from_graph += (end - start)

        start = time.time()
        adj = adj.to_dense()
        end = time.time()
        global adj_todense
        adj_todense += (end - start)

        start = time.time()
        norm = g.ndata['norm']
        end = time.time()
        global get_norm
        get_norm += (end - start)

        start = time.time()
        norm = norm.unsqueeze(1)
        #norm_2 = norm.unsqueeze(0)
        end = time.time()
        global norm_transform
        norm_transform += (end - start)


        start = time.time()
        x = self.conv1(adj, features, norm)
        end = time.time()
        global conv1
        conv1 += (end - start)

        start = time.time()
        x = F.relu(x)
        end = time.time()
        global relu
        relu += (end - start)

        start = time.time()
        x = self.conv2(adj, x, norm)
        end = time.time()
        global conv2
        conv2 += (end - start)


        start = time.time()

        g.ndata['h'] = x
        meannode = dgl.mean_nodes(g, 'h')

        end = time.time()
        global meanNode_time
        meanNode_time += (end - start)
        
        return meannode

model = DenseGCN(4, 16, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
state_dict = torch.load('../model_parameters/AIDS_model.pth')
model.load_state_dict(state_dict)
model.eval()

num_correct = 0
num_tests = 0
elapsed_time = 0
test_num = 0

for batched_graph, labels in test_dataloader:
    print("test_num: ", test_num)
    test_num += 1
    # print("batched_graph: ",batched_graph)
    length = len(labels)
    print("-----------------------------------------------------")
    print("labels: ",labels)
    print("labels.squeeze(): ",labels.squeeze())
    labels = labels.squeeze()
    start = time.time()
    pred = model(batched_graph, batched_graph.ndata["node_attr"].float())
    end = time.time()
    elapsed_time = elapsed_time + (end - start)

    print("pred: ", pred)
    print("pred.argmax(1): ", pred.argmax(1))
    print("pred.argmax(1).squeeze(): ", pred.argmax(1).squeeze())
    print("(pred.argmax(1)==labels).sum: ", (pred.argmax(1) == labels).sum())
    print("(pred.argmax(1)==labels).sum.item(): ", (pred.argmax(1) == labels).sum().item())

    print("num_correct_before:", num_correct)
    num_correct += (pred.argmax(1) == labels).sum().item()
    print("num_correct_after:", num_correct)
    num_tests += length

print("---------------------------------------------------------")
print("num_correct: ",num_correct)
print("num_tests: ",num_tests)
print("Test accuracy:", num_correct / num_tests)
print("Total_time:", elapsed_time)

print("---------------------------------------------------------")
print("adj_tonumpy_1_time: ", adj_tonumpy_1_time)
print("adj_tonumpy_2_time: ", adj_tonumpy_2_time)
print("Htonumpy_1_time: ", Htonumpy_1_time)
print("Htonumpy_2_time: ", Htonumpy_2_time)
print("Wtonumpy_1_time: ", Wtonumpy_1_time)
print("Wtonumpy_2_time: ", Wtonumpy_2_time)
print("KeyMM_1_time: ", KeyMM_1_time)
print("KeyMM_2_time: ", KeyMM_2_time)
print("linear_1_time: ", linear_1_time)
print("linear_2_time: ", linear_2_time)
print("bias_1_time: ", bias_1_time)
print("bias_2_time: ", bias_2_time)
print("torch_transform_1_time : ", torch_transform_1_time )
print("torch_transform_2_time : ", torch_transform_2_time )


print("---------------------------------------------------------")
print("get_adj_from_graph_time: ", get_adj_from_graph)
print("adj_todense_time: ", adj_todense)
print("get_norm_time: ", get_norm)
print("norm_transform_time: ", norm_transform)
print("DAD_time: ", DAD_time)
print("---------------------------------------------------------")
print("Normalization_1_1_time: ",Normalize_1_1_time)
print("Normalization_1_2_time: ",Normalize_1_2_time)
print("Normalization_2_1_time: ",Normalize_2_1_time)
print("Normalization_2_2_time: ",Normalize_2_2_time)
print("conv1_time: ",conv1)
print("relu_time: ",relu)
print("conv2_time: ",conv2)
print("meanNode_time: ",meanNode_time)
print("---------------------------------------------------------")
print("Infer time:", DAD_time + conv1 + conv2 + relu)

