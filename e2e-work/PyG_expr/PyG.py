import torch
from torch_geometric.datasets import TUDataset
import time
dataset = TUDataset('data/TUDataset', name = 'DHFR')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:600]
test_dataset = dataset[600:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)



from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats,num_classes)
        #self.conv3 = GCNConv(hidden_channels, hidden_channels)
        #self.lin = Linear(h_feats, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 1. 获得节点嵌入
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        '''
        x = x.relu()
        x = self.conv3(x, edge_index)
        '''
        
        
        
        return x

model = GCN(in_feats = dataset.num_node_features, h_feats = 16, num_classes = dataset.num_classes)
model.load_state_dict(torch.load('DHFR_model.pth'))
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    
    for data in train_loader:
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    
    correct = 0
    elapsed_time = 0
    for data in loader:                            # 批遍历测试集数据集。
        start = time.time()
        out = model(data.x, data.edge_index, data.batch) # 一次前向传播
        end = time.time()
        elapsed_time += end - start
        pred = out.argmax(dim=1)                         # 使用概率最高的类别
        correct += int((pred == data.y).sum())           # 检查真实标签
        
    print("This time Inference time is:",elapsed_time)
    return correct / len(loader.dataset)
test_acc = test(test_loader)
print("acc: ",test_acc)

'''
for epoch in range(1, 121):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
'''    

