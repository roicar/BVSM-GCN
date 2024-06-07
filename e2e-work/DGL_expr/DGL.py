

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import timeit
import statistics

dataset = dgl.data.TUDataset("Letter-low")
# 1. 读取数据集

# 2. 获取图的节点数并存储（图的索引，节点数）
graph_sizes = [(i, g[0].num_nodes()) for i, g in enumerate(dataset)]

# 3. 根据节点数排序
sorted_indices = [i for i, _ in sorted(graph_sizes, key=lambda x: x[1])]

# 4. 创建一个新的数据集列表，按照排序后的索引
dataset = [dataset[i] for i in sorted_indices]



from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

'''
num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=32, drop_last=False
)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=32, drop_last=False
)
'''

from torch.utils.data.sampler import SequentialSampler
num_examples = len(dataset)
# num_train = int(num_examples * 0.8)
num_test = int(num_examples)

# train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SequentialSampler(torch.arange(num_test))

# train_dataloader = GraphDataLoader(
#    dataset, sampler=train_sampler, batch_size=1, drop_last=False
# )

# Here we compare with BVSM-M, batch_size=num_test
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=1, drop_last=False
)


######################################################################
# You can try to iterate over the created ``GraphDataLoader`` and see what it
# gives:
#


it = iter(test_dataloader)
batch = next(it)
print("batch: ",batch)




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
print(graphs)


######################################################################
# Define Model
# ------------
#
# This tutorial will build a two-layer `Graph Convolutional Network
# (GCN) <http://tkipf.github.io/graph-convolutional-networks/>`__. Each of
# its layer computes new node representations by aggregating neighbor
# information. If you have gone through the
# :doc:`introduction <1_introduction>`, you will notice two
# differences:
#
# -  Since the task is to predict a single category for the *entire graph*
#    instead of for every node, you will need to aggregate the
#    representations of all the nodes and potentially the edges to form a
#    graph-level representation. Such process is more commonly referred as
#    a *readout*. A simple choice is to average the node features of a
#    graph with ``dgl.mean_nodes()``.
#
# -  The input graph to the model will be a batched graph yielded by the
#    ``GraphDataLoader``. The readout functions provided by DGL can handle
#    batched graphs so that they will return one representation for each
#    minibatch element.
#

from dgl.nn.pytorch.conv import GraphConv

conv1 = 0
relu = 0
conv2 = 0
meanTime = 0
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats,allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes,allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        start = time.time()
        h = self.conv1(g, in_feat)
        end = time.time()
        global conv1
        conv1 += end-start

        start = time.time()
        h = F.relu(h)
        end = time.time()
        global relu
        relu += end-start

        start = time.time()
        h = self.conv2(g, h)
        end = time.time()
        global conv2
        conv2 += end-start

        
        
        start = time.time()
        g.ndata["h"] = h
        mean_node = dgl.mean_nodes(g, "h")
        end = time.time()
        global meanTime
        meanTime += (end-start)
        return mean_node


######################################################################
# Training Loop
# -------------
#
# The training loop iterates over the training set with the
# ``GraphDataLoader`` object and computes the gradients, just like
# image classification or language modeling.
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using Device:", device)
# Create the model with given dimensions
model = GCN(2, 16, 15)
#model = model.to(device)
print("OK")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print("OK2")

# for epoch in range(1000):
#     for batched_graph, labels in train_dataloader:
#         pred = model(batched_graph, batched_graph.ndata["node_attr"].float())
#         #print("batched_graph: ",batched_graph)
#         #print("batched_graph.ndata[attr] ", batched_graph.ndata["node_attr"].float())
#         #print("pred: ",pred)
#         #print("label: ",labels)
#         loss = F.cross_entropy(pred, labels.squeeze(dim=1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch (epoch):loss {loss.item()}')
#
# #save parameters
# torch.save(model.state_dict(),'Letter_low_model.pth')


model.load_state_dict(torch.load('../model_parameters/Letter-low_model.pth'))

num_correct = 0
num_tests = 0
elapsed_time = 0
# MinexeTime = 0
# MaxexeTime = 0
# MedianTime = 0
for batched_graph, labels in test_dataloader:
    #batched_graph = batched_graph.to(device)
    #labels = labels.to(device)
    #print("batched_graph: ",batched_graph)
    length = len(labels)
    #print("labels: ",labels)
    #print("labels.squeeze(): ",labels.squeeze())
    labels = labels.squeeze()
    # exe_time = []
    # for _ in range(1500):
    start = time.time()
    pred = model(batched_graph, batched_graph.ndata["node_attr"].float())
    end = time.time()
    elapsed_time += end-start
        # exe_time.append(end - start)

    # MinexeTime += min(exe_time)
    # MaxexeTime += max(exe_time)
    # MedianTime += statistics.median(exe_time)
    #print("pred: ",pred)
    #print("pred.argmax(1): ",pred.argmax(1))
    #print("pred.argmax(1).squeeze(): ",pred.argmax(1).squeeze())
    #print("(pred.argmax(1)==labels).sum: ",(pred.argmax(1) == labels).sum())
    #print("(pred.argmax(1)==labels).sum.item(): ",(pred.argmax(1) == labels).sum().item())
    #print("num_correct_before:",num_correct)
    num_correct += (pred.argmax(1) == labels).sum().item()
    #print("num_correct_after:",num_correct)
    num_tests += length
    
print("conv1 time: ",conv1)
print("relu time: ",relu)
print("conv2 time: ",conv2)
print("mean time: ",meanTime)
# print("MaxTime: ",MaxexeTime)
# print("MinTime: ",MinexeTime)
# print("MedianTime: ",MedianTime)
print("num_correct: ",num_correct)
print("num_tests: ",num_tests)
print("Test accuracy:", num_correct / num_tests)
print("Infer time:", elapsed_time)


