import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros

class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 将稀疏矩阵乘法转化为稠密矩阵乘法
        support = torch.mm(x, self.weight)
        output = torch.zeros_like(support)
        for i in range(edge_index.size(1)):
            output[row[i]] += norm[i] * support[col[i]]
        output += self.bias
        return output

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def main():
    dataset = TUDataset(root='/tmp/TUDataset', name='PROTEINS')
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = GCN(dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    for epoch in range(200):
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            print(out)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

    model.eval()
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct = float(pred.eq(data.y).sum().item())
        acc = correct / data.num_nodes
        print('Accuracy: {:.4f}'.format(acc))
    '''
    model.train()
    for epoch in range(200):
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

    model.eval()
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct = float(pred.eq(data.y).sum().item())
        acc = correct / len(data.y)
        print('Accuracy: {:.4f}'.format(acc))
    '''

if __name__ == "__main__":
    main()




