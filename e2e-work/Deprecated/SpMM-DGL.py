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
#print("dataset:",dataset)
def preprocess_graphs(graphs):
    for i,g in enumerate(graphs):
        # add selfloop
        print("g:",g)
        g = dgl.add_self_loop(g[0])
        
        # normalization
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs,-0.5)
        norm = norm.to(g.device).unsqueeze(1)
        
        # store the normalization factoe in edge for GCN
        print("OK1")
        print("g before:",g)
        g.edata['norm'] = norm[g.edges()[0]] * norm[g.edges()[1]]
        print("OK2")
        print("g after",g)
        graphs = list(graphs)
        graphs[i] = g
        graphs = tuple(graphs)
        
    return graphs

dataset = preprocess_graphs(dataset)
print("dataset:",dataset)
print("Max number of node:", dataset.max_num_node)
print("Number of graph categories:", dataset.num_classes)


from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#from torch.utils.data.sampler import SequentialSampler

'''
num_examples = len(dataset)
num_train = int(num_examples * 0.8)
num_test = num_examples - num_train

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=32, drop_last=False
)

test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=1, drop_last=False
)


'''

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
'''
class SparseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SparseGCNConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
'''
'''
    def forward(self, adj, features):
    
        start = time.time()
        sparse_feat = torch.sparse.mm(adj, features)
        end = time.time()
        global DMM_time
        DMM_time += (end - start)
        return self.linear(sparse_feat)
'''
'''
    def forward(self, g, features):
        # 获取归一化因子
        norm = g[0].edata['norm']
        
        # 应用归一化因子到特征上
        g.ndata['h'] = features
        g.apply_edges(lambda edges: {'norm_h': edges.data['norm'] * edges.src['h']})
        
        # 聚合邻居特征
        g.update_all(message_func=dgl.function.copy_e('norm_h', 'm'),
                     reduce_func=dgl.function.sum('m', 'h_neigh'))
        
        # 获取聚合后的邻居特征
        h_neigh = g.ndata.pop('h_neigh')
        
        # 线性变换
        result = self.linear(h_neigh)
        return result
class SparseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SparseGCN, self).__init__()
        self.conv1 = SparseGCNConv(in_feats, h_feats)
        self.conv2 = SparseGCNConv(h_feats, num_classes)

    def forward(self, g, features):
'''
'''
        start = time.time()
        adj = g.adjacency_matrix()
        adj = torch.eye(g.number_of_nodes()).to(adj.device) + adj
        deg = adj.sum(1)
        d_inv_sqrt = torch.pow(deg, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        end = time.time()
        global time1
        time1 += (end-start)
'''
'''
        adj = g.adjacency_matrix()
        #start = time.time()
        x = F.relu(self.conv1(adj, features))
        #end = time.time()
        #global time2
        #time2 += (end-start)
        
        #start = time.time()
        x = self.conv2(adj_normalized, x)
        #end = time.time()
        #global time3
        #time3 += (end-start)
        
        #start = time.time()
        g.ndata['h'] = x
        meannode = dgl.mean_nodes(g, 'h')
        #end = time.time()
        #global time4
        #time4 += (end-start)
        
        return meannode
'''
class SparseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SparseGCNConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, features):
        # Ensure 'norm' is present in edge data
        print("g: ",g)
        if 'norm' not in g.edata:
            raise ValueError("Graph does not have 'norm' in edge data.")
        
        # Apply normalization factor to features
        g.ndata['h'] = features
        g.apply_edges(lambda edges: {'norm_h': edges.data['norm'] * edges.src['h']})
        
        # Aggregate neighbor features
        g.update_all(message_func=dgl.function.copy_e('norm_h', 'm'),
                     reduce_func=dgl.function.sum('m', 'h_neigh'))
        
        # Get aggregated neighbor features
        h_neigh = g.ndata.pop('h_neigh')
        
        # Linear transformation
        result = self.linear(h_neigh)
        return result

class SparseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SparseGCN, self).__init__()
        self.conv1 = SparseGCNConv(in_feats, h_feats)
        self.conv2 = SparseGCNConv(h_feats, num_classes)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        
        # Assuming 'h' is the feature you want to aggregate at the end
        g.ndata['h'] = x
        mean_node = dgl.mean_nodes(g, 'h')
        
        return mean_node
# Initialize the model
model = SparseGCN(3, 16, dataset.num_classes)



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
state_dict = torch.load('COX2_model_SpMM_new.pth')
#print("state_dict: ",state_dict)
'''
new_state_dict = {}

# Update the keys
for key in state_dict:
    new_key = key.replace("conv1.", "conv1.linear.").replace("conv2.", "conv2.linear.")
    new_state_dict[new_key] = state_dict[key]
'''
model.load_state_dict(state_dict)

model.eval()

num_correct = 0
num_tests = 0
elapsed_time = 0
test_num = 0
for batched_graph, labels in test_dataloader:
    print("test_num: ",test_num)
    test_num += 1
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

