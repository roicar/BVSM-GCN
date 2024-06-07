import torch
import numpy as np
import dgl
from torch.nn.functional import pad
from dgl.data import TUDataset
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import time

# 预处理函数，填充并合并图
def batch_graphs(graphs):
    for g in graphs:
        print("OK")
        print("g[0]:",g[0])
    max_nodes = max(g[0].num_nodes for g in graphs)
    batch_features = []
    batch_adj = []

    for g, _ in graphs:
        features = g[0].ndata['node_attr'].float()  # 假设特征存储在 'node_attr'
        adj_matrix = g[0].adjacency_matrix().to_dense()

        num_nodes = g[0].number_of_nodes()
        pad_size = max_nodes - num_nodes

        padded_features = pad(features, (0, 0, 0, pad_size))
        padded_adj = pad(adj_matrix, (0, pad_size, 0, pad_size))

        batch_features.append(padded_features)
        batch_adj.append(padded_adj)

    batch_features = torch.stack(batch_features)
    batch_adj = torch.stack(batch_adj)

    return batch_features, batch_adj

# GCN模型定义
class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

# 加载数据集和模型
dataset = TUDataset("BZR")
model = GCN(in_feats=3, h_feats=16, num_classes=2)  # 参数按需调整

# 假定有预训练模型
model.load_state_dict(torch.load('./model_parameters/BZR_model.pth'))

# 转换模型参数格式
params = {k: tvm.nd.array(v.detach().numpy()) for k, v in model.state_dict().items()}

# 准备批量数据
batch_size = 5  # 示例批量大小，根据需求调整
batched_graphs = batch_graphs(dataset[:batch_size])

# 模型输入维度调整，为了符合TVM的要求
features = batched_graphs[0] # 示例数据，应替换为batched_graphs[0]
adj = batched_graphs[1]  # 示例数据，应替换为batched_graphs[1]

# 构建Relay图
input_name = "input"
shape_dict = {input_name: features.shape}
mod, params = relay.frontend.from_pytorch(model, shape_dict)

# 编译模型
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build_module.build(mod, target, params=params)

# 创建图执行器
ctx = tvm.device(str(target), 0)
m = graph_executor.create(graph, lib, ctx)

# 设置输入并运行
m.set_input(input_name, tvm.nd.array(batched_graphs[0].numpy()))
m.set_input('adj', tvm.nd.array(batched_graphs[1].numpy()))
m.set_input(**params)
m.run()

# 获取输出
tvm_output = m.get_output(0).asnumpy()


