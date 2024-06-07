"""
Here we can infer GCN with DGL+Pytorch
We keep the train module in the front

"""

import os

import numpy as np

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import timeit
import statistics

#Notice:You need to change parameter of infeat_nums after changing dataset
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
maxnodes = dataset.max_num_node
numclasses = dataset.num_classes
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
        norm = norm.to(g.device).unsqueeze(1)
        #print(g)
        g.ndata['norm'] = norm
        processed_graphs.append((g,t))
    return processed_graphs

dataset = preprocess_graphs(inference_subset)

print("Max number of node:", maxnodes)
print("Number of graph categories:", numclasses)


from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#from torch.utils.data.sampler import SequentialSampler






from torch.utils.data.sampler import SequentialSampler
num_examples = len(dataset)
#num_train = int(num_examples * 0.01)
num_test = int(num_examples )

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
#print(graphs)





DAD_time = 0
OtherNormalization_time = 0
KeyMM_time = 0
MM_time = 0
meanNode_time = 0
init_time = 0

class SparseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        time_7 = time.time()
        super(SparseGCNConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        time_8 = time.time()
        global init_time
        init_time += (time_8 - time_7)
    
    def forward(self, adj, features):
    
        start = time.time()
        sparse_feat = torch.sparse.mm(adj, features)
        end = time.time()
        global KeyMM_time
        KeyMM_time += (end - start)
        #print("This graph sparseMM time is:",(end - start))
        return self.linear(sparse_feat)
    
    
class SparseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SparseGCN, self).__init__()
        self.conv1 = SparseGCNConv(in_feats, h_feats)
        self.conv2 = SparseGCNConv(h_feats, num_classes)
        
        
    def normalize_adjacency(self, g):
        time_1 = time.time()
        adj = g.adjacency_matrix()
        adj = adj.to_dense()
        norm = g.ndata['norm']
        norm_t = np.squeeze(norm, axis=1)

        time_2 = time.time()
        global OtherNormalization_time
        OtherNormalization_time += (time_2 - time_1)
        
        time_3 = time.time()
        adj = adj * norm
        adj_normalized = adj * norm_t
        time_4 = time.time()
        global DAD_time
        DAD_time += (time_4 - time_3)
        return adj_normalized


    def forward(self, g, features):
        #start = time.time()        
        adj_normalized = self.normalize_adjacency(g)
        
        
        time_5 = time.time()
        x = F.relu(self.conv1(adj_normalized, features))
        x = self.conv2(adj_normalized, x)
        time_6 = time.time()
        global MM_time
        MM_time += (time_6 - time_5)
        
        start = time.time()
        '''
        max_nodes = adj_normalized.shape[1]
        x = x.sum(dim=1) / max_nodes
        '''
        g.ndata['h'] = x
        meannode = dgl.mean_nodes(g, 'h')

        end = time.time()
        global meanNode_time
        meanNode_time += (end-start)
        
        return meannode

# Initialize the model
model = SparseGCN(4, 16, numclasses)



# Create the model with given dimensions
#print("OK7")
#model = DenseGCN(3, 16, dataset.num_classes)
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
torch.save(model.state_dict(),'DHFR_model_SpMM_new.pth')
# pass the parameters
'''

#state_dict = torch.load('BZR_model_Dense.pth')
state_dict = torch.load('AIDS_model.pth')
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
MinexeTime = 0
MaxexeTime = 0
MedianTime = 0
for batched_graph, labels in test_dataloader:
    print("test_num: ",test_num)
    test_num += 1
    #print("batched_graph: ",batched_graph)
    length = len(labels)
    print("-----------------------------------------------------")
    #print("labels: ",labels)
    #print("labels.squeeze(): ",labels.squeeze())
    labels = labels.squeeze()
    exe_time = []
    for _ in range(1000):
        start = time.time()
        pred = model(batched_graph, batched_graph.ndata["node_attr"].float())
        end = time.time()
        exe_time.append(end - start)


    MinexeTime += min(exe_time)
    MaxexeTime += max(exe_time)
    MedianTime += statistics.median(exe_time)



    print("pred: ",pred)
    print("pred.argmax(1): ",pred.argmax(1))
    print("pred.argmax(1).squeeze(): ",pred.argmax(1).squeeze())
    print("(pred.argmax(1)==labels).sum: ",(pred.argmax(1) == labels).sum())
    print("(pred.argmax(1)==labels).sum.item(): ",(pred.argmax(1) == labels).sum().item())


    print("num_correct_before:",num_correct)
    num_correct += (pred.argmax(1) == labels).sum().item()
    print("num_correct_after:",num_correct)
    num_tests += length
    
print("DAD_time: ", DAD_time)
print("OtherNormalization_time: ",OtherNormalization_time)
print("MM_time: ",MM_time)
print("KeyMM_time: ",KeyMM_time)
print("meanNode_time: ",meanNode_time)
print("init_time: ", init_time)

print("MaxTime: ",MaxexeTime)
print("MinTime: ",MinexeTime)
print("MedianTime: ",MedianTime)

print("num_correct: ",num_correct)
print("num_tests: ",num_tests)
print("Test accuracy:", num_correct / num_tests)
# print("Infer time:", elapsed_time)



