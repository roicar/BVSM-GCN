import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import numpy as np
import statistics
import sys

def showDensity(DS):
    #Notice:You need to change parameter of infeat_nums after changing dataset
    #dataset = dgl.data.TUDataset("AIDS")
    #dataset = dgl.data.TUDataset("BZR")
    #dataset = dgl.data.TUDataset("COX2")
    #dataset = dgl.data.TUDataset("DHFR")
    dataset = dgl.data.TUDataset(DS)
    # 1. 读取COX2数据集

    # 2. 获取图的节点数并存储（图的索引，节点数）
    graph_sizes = [(i, g[0].num_nodes()) for i, g in enumerate(dataset)]

    # 3. 根据节点数排序
    sorted_indices = [i for i, _ in sorted(graph_sizes, key=lambda x: x[1])]

    # 4. 创建一个新的数据集列表，按照排序后的索引
    sorted_dataset = [dataset[i] for i in sorted_indices]
    i = 1
    average_density = 0
    for g in sorted_dataset:
        g = dgl.add_self_loop(g[0])
        adj = g.adjacency_matrix().to_dense()
        print("adj:",adj)
        deg = adj.sum(1)
        d_inv_sqrt = torch.pow(deg, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        adj = adj * d_inv_sqrt.unsqueeze(1)
        adj_normalized = adj * d_inv_sqrt.unsqueeze(0)
        density = np.count_nonzero(adj_normalized)/(adj_normalized.shape[0]*adj_normalized.shape[1])
        print("{} Graph density: ".format(i), density)
        average_density += density
        i = i+1

    print("Average density: ", average_density/(i-1))

if __name__ == '__main__':

    DS = str(sys.argv[1])
    showDensity(DS)