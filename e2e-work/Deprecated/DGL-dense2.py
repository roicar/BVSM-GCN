import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class DenseGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(DenseGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1) ** 0.5
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, adj, features):
        # 稠密矩阵乘法实现图卷积
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        return output

class DenseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(DenseGCN, self).__init__()
        self.gc1 = DenseGCNLayer(in_feats, h_feats)
        self.gc2 = DenseGCNLayer(h_feats, num_classes)

    def forward(self, g, features):
        # 将图转换为邻接矩阵（稠密形式）
        adj = g.adjacency_matrix().to_dense()
        x = F.relu(self.gc1(adj, features))
        x = self.gc2(adj, x)
        return x

# 假设模型训练和保存代码与之前相同
