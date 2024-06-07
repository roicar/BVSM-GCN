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


#Notice:You need to change parameter of infeat_nums after changing dataset
#dataset = dgl.data.TUDataset("AIDS")
#dataset = dgl.data.TUDataset("BZR")
#dataset = dgl.data.TUDataset("COX2")
#dataset = dgl.data.TUDataset("DHFR")
dataset = dgl.data.TUDataset("COX2")


print("Max number of node:", dataset.max_num_node)
print("Number of graph categories:", dataset.num_classes)


from dgl.dataloading import GraphDataLoader

from torch.utils.data.sampler import SubsetRandomSampler


num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

def collate(samples):
    # `samples` 是一个列表，其中包含图和标签
    graphs, labels = map(list, zip(*samples))
    # 根据节点数量对图进行分组
    groups = {}
    for graph, label in zip(graphs, labels):
        num_nodes = graph.number_of_nodes()
        if num_nodes not in groups:
            groups[num_nodes] = ([], [])
        groups[num_nodes][0].append(graph)
        groups[num_nodes][1].append(label)
    # 对每个组进行批处理
    batched_graphs = {}
    for num_nodes, (gs, ls) in groups.items():
        batched_graphs[num_nodes] = (dgl.batch(gs), torch.tensor(ls))
    return batched_graphs, labels

# 使用自定义的批处理函数
train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=5, drop_last=False, collate_fn=collate)
test_dataloader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=5, drop_last=False, collate_fn=collate)

######################################################################
# You can try to iterate over the created ``GraphDataLoader`` and see what it
# gives:
#


it = iter(train_dataloader)
batch = next(it)
print(batch)




batched_graph, labels = batch
'''
print(
    "Number of nodes for each graph element in the batch:",
    batched_graph.batch_num_nodes(),
)
print(
    "Number of edges for each graph element in the batch:",
    batched_graph.batch_num_edges(),
)
'''
# Recover the original graph elements from the minibatch
#graphs = dgl.unbatch(batched_graph)
#print("The original graphs in the minibatch:")
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
class DenseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(DenseGCNConv, self).__init__()
        #self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        #self.reset_parameters()
        #print("OK1")
        self.linear = nn.Linear(in_feats,out_feats)
        #print("OK2")

    #def reset_parameters(self):
        #nn.init.xavier_uniform_(self.weight)

    def forward(self, adj, features):
        #print("OK3")
        dense_adj = adj.to_dense()
        #print("shape: ",dense_adj.shape())
        dense_feat = torch.matmul(dense_adj,features)
        #print("OK4")
        #support = torch.mm(input_features, self.weight)
        #output = torch.mm(adjacency_matrix, support)
        return self.linear(dense_feat)




#from dgl.nn import GraphConv


class DenseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(DenseGCN, self).__init__()
        print("in_feats: ",in_feats)
        print("h_feats: ",h_feats)
        self.conv1 = DenseGCNConv(in_feats,h_feats)
        self.conv2 = DenseGCNConv(h_feats,num_classes)
        #print("OK5")
        
    def forward(self, batched_graphs, features):
        # `batched_graphs` 是一个字典，键是节点数，值是批处理的图
        outputs = []
        bit = batched_graphs.items()
        print("list(bit): ",list(bit))
        for num_nodes, (bg, _) in list(bit):
            adj = bg.adjacency_matrix().to_dense() + torch.eye(num_nodes)
            deg_inv_sqrt = adj.sum(dim=1).pow(-0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
            deg_inv_sqrt = torch.diag(deg_inv_sqrt)
            adj_normalized = torch.matmul(torch.matmul(deg_inv_sqrt, adj), deg_inv_sqrt)
            
            x = F.relu(self.conv1(adj_normalized, features))
            x = self.conv2(adj_normalized, x)
            bg.ndata['h'] = x
            outputs.append(dgl.mean_nodes(bg, 'h'))
        # 合并不同维度的输出
        return torch.cat(outputs, dim=0)




# Create the model with given dimensions
#print("OK7")
model = DenseGCN(3, 16, dataset.num_classes)
#print("OK8")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


##########################################################################
# Training Loop
# Skip it when you need inference
'''
for epoch in range(75):
    for batched_graphs, labels in train_dataloader:
        print("list(batched_graphs.values())[0]: ",list(batched_graphs.values())[0])
        outputs = model(batched_graphs, list(batched_graphs.values())[0][0].ndata["node_attr"].float())
        #print("batched_graph: ",batched_graph)
        #print("batched_graph.ndata[attr] ", batched_graph.ndata["node_attr"].float())
        #print("pred: ",pred)
        #print("label: ",labels)
        loss = F.cross_entropy(outputs, labels.squeeze(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch (epoch):loss {loss.item()}')

#save parameters
torch.save(model.state_dict(),'BZR_DGL_B.pth')
'''

# pass the parameters

state_dict = torch.load('COX2_model.pth')

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
for batched_graph, labels in test_dataloader:
    #print("batched_graph: ",batched_graph)
    length = len(labels)
    print("-----------------------------------------------------")
    print("labels: ",labels)
    #print("labels.squeeze(): ",labels.squeeze())
    #labels = labels.squeeze()
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
print("num_correct: ",num_correct)
print("num_tests: ",num_tests)
print("Test accuracy:", num_correct / num_tests)
print("Infer time:", elapsed_time)

