import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import dgl.data

os.environ["DGLBACKEND"] = "pytorch"

# Load dataset
dataset = dgl.data.TUDataset("COX2")

print("Max number of nodes:", dataset.max_num_node)
print("Number of graph categories:", dataset.num_classes)

max_nodes = dataset.max_num_node
num_features = 3

# Dense GCN Convolution Layer
class DenseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(DenseGCNConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, features):
        return self.linear(features)

# Dense GCN Model
class DenseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(DenseGCN, self).__init__()
        self.conv1 = DenseGCNConv(in_feats, h_feats)
        self.conv2 = DenseGCNConv(h_feats, num_classes)

    def forward(self, features):
        x = F.relu(self.conv1(features))
        x = self.conv2(x)
        return x

# Model, Loss, and Optimizer
model = DenseGCN(num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

# DataLoader
data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

# Training Loop
'''
for epoch in range(75):
    for batch in data_loader:
        graphs, labels = batch
        features = torch.stack([g.ndata['node_attr'] for g in graphs])
        labels = labels.long()
        
        # Forward pass
        pred = model(features)
        loss = loss_func(pred, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss {loss.item()}')

# Save Model
torch.save(model.state_dict(), 'COX2_DenseGCN.pth')
'''
state_dict = torch.load('COX2_model.pth')
# Test Model
model.eval()
num_correct = 0
num_tests = 0
for batch in data_loader:
    graphs, labels = batch
    features = torch.stack([g.ndata['node_attr'] for g in graphs])
    labels = labels.long()
    
    pred = model(features)
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += labels.size(0)

print("Test accuracy:", num_correct / num_tests)
