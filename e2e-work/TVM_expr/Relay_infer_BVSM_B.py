"""
Building a Graph Convolutional Network combined with BVSM
======================================
**Author**: Daiwen

Tasks:
1.construct a simple GNN(GCN) as same as that in PyG/DGL.
2.load parameters into model

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
import time
from dgl.data import load_data
from dgl.nn.pytorch import GraphConv
import math
import statistics
import sys
from tvm import relay
from tvm.contrib import graph_executor
import tvm
import torch
import numpy as np


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats,allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes,allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")




def evaluate(graphs, logits):
    correct = 0
    pred = logits.mean(1).argmax(1)
    #print("pred:",pred)
    #print("pred[0].item():",pred[0].item())
    for i,(g, label) in enumerate(graphs):
        if pred[i].item() == label.item():
            correct += 1

    return correct




import numpy as np
import networkx as nx

def prepare_params(graphs):

    batch_size = len(graphs)
    # 初始化填充后的特征和邻接矩阵的列表
    features_list = []
    adj_list = []
    for g, _ in graphs:
        features = g.ndata['node_attr'].float()  # 确保特征是浮点类型
        adj_matrix = g.adjacency_matrix().to_dense()
        diagonal_unit_matrix = np.eye(adj_matrix.shape[0], dtype=np.float32)
        adj_matrix = adj_matrix + diagonal_unit_matrix  # 添加邻接矩阵
        # 计算填充大小
        num_nodes = g.number_of_nodes()




        features_list.append(features)
        adj_list.append(adj_matrix)

    batch_features = torch.stack(features_list).numpy()
    batch_adj = torch.stack(adj_list).numpy()
    params = {}
    params["infeats"] = batch_features
    params["adj"] = batch_adj

    # Normalization w.r.t. node degrees
    degs = [g.in_degrees(i) for i in range(g.number_of_nodes())]
    degs = np.clip(degs, 1, None)  # 限制度在1以上
    params["norm"] = np.power(degs, -0.5).astype("float32")
    params["norm"] = np.clip(params["norm"],0,1)
    params["norm1"] = np.stack([params["norm"].reshape((params["norm"].shape[0], 1))]*batch_size)
    params["norm2"] = np.stack([params["norm"].reshape((1, params["norm"].shape[0]))]*batch_size)
    return params

def Graphconv(layer_name, input_dim, output_dim, adj, input, bias=True, activation=None, batch_size=1):
    start1 = time.time()
    start = time.time()
    weight = relay.var(layer_name + ".weight", shape=(input_dim, output_dim))
    end = time.time()
    global CreateW_time
    CreateW_time += (end - start)


    start = time.time()
    weight = relay.op.transform.expand_dims(weight, axis=0, num_newaxis=1)
    end = time.time()
    global TransformW_time
    TransformW_time += (end - start)


    start = time.time()
    dense_1 = relay.nn.batch_matmul(adj,input,transpose_b=False)
    end = time.time()
    if output_dim == 16:
        global AH_1_time
        AH_1_time += (end - start)
    else:
        global AH_2_time
        AH_2_time += (end - start)


    start = time.time()
    output = relay.nn.batch_matmul(dense_1, weight, transpose_b=False)
    end = time.time()
    if output_dim == 16:
        global HW_1_time
        HW_1_time += (end - start)
    else:
        global HW_2_time
        HW_2_time += (end - start)


    # Only record the first layer time

    if bias is True:
        start = time.time()
        _bias = relay.var(layer_name + ".bias", shape=(output_dim, ))
        end = time.time()
        global bias_var
        bias_var += (end - start)

        start = time.time()
        output = relay.nn.bias_add(output, _bias, axis=-1)
        end = time.time()
        global biasAdd_time
        biasAdd_time += (end - start)

    if activation is not None:
        start = time.time()
        output = activation(output)
        end = time.time()
        global activation_time
        activation_time += (end - start)
    return output


def process_batch(batch_graphs,sorted_dataset):
    dgl_g = sorted_dataset[0]
    # print("dgl_g: ",dgl_g)
    # print("dgl_g[0]: ",dgl_g[0])
    num_layers = 1
    num_hidden = 16
    features = dgl_g[0].ndata['node_attr']
    features = features.float()
    infeat_dim = features.shape[1]
    num_classes = 4

    torch_model = GCN(in_feats=infeat_dim, h_feats=num_hidden, num_classes=num_classes)
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
    torch_model.load_state_dict(torch.load(DS + '_model.pth'))

    right_rel = 0.0
    elapsed_Totaltime = 0
    num = 0.0
    from tvm import relay
    from tvm.contrib import graph_executor
    import tvm
    from tvm import te

    batch_size = len(batch_graphs)
    params = prepare_params(batch_graphs)
    infeats = relay.var("infeats", shape=params["infeats"].shape)
    norm1 = relay.Constant(tvm.nd.array(params["norm1"]))
    norm2 = relay.Constant(tvm.nd.array(params["norm2"]))
    adj = relay.const(tvm.nd.array(params["adj"]))
    # print("OK1")
    adj = relay.multiply(adj, norm1)
    # print("OK2")
    adj = relay.multiply(norm2, adj)
    print("adj:", adj)
    # Construct the 2-layer GCN
    layers = []
    layers.append(
        Graphconv(
            #layer_name="layers.0",
            layer_name="conv1",
            input_dim=infeat_dim,
            output_dim=num_hidden,
            adj=adj,
            input=infeats,
            activation=relay.nn.relu,
            batch_size=batch_size,
        )
    )
    layers.append(
        Graphconv(
            #layer_name="layers.1",
            layer_name="conv2",
            input_dim=num_hidden,
            output_dim=num_classes,
            adj=adj,
            input=layers[-1],
            activation=None,
            batch_size=batch_size,
        )
    )

    # Analyze free variables and generate Relay function
    output = layers[-1]
    #print("shape:",output.shape())
    # Compile and run with TVM
    model_params = {}
    for param_tensor in torch_model.state_dict():
        model_params[param_tensor] = torch_model.state_dict()[param_tensor].numpy()

    for i in range(num_layers + 1):
        params["conv%d.weight" % (i+1)] = model_params["conv%d.weight" % (i+1)]
        params["conv%d.bias" % (i+1)] = model_params["conv%d.bias" % (i+1)]
    # Set the TVM build target
    #target = "cuda"  # Currently only support `llvm` as target
    target ='llvm'
    print("OK1")
    func = relay.Function(relay.analysis.free_vars(output), output)
    func = relay.build_module.bind_params_by_name(func, params)
    mod = tvm.IRModule()
    mod["main"] = func
    # Build with Relay
    print("OK2")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    print("OK3")
    # Generate graph executor
    #dev = tvm.cuda()
    dev = tvm.device(target, 0)
    m = graph_executor.GraphModule(lib["default"](dev))
    print("OK4")
    exe_time = []
    for _ in range(1000):
        start = time.time()
        m.run()
        end = time.time()
        exe_time.append(end - start)

    print(f"min time is {min(exe_time)}")
    print(f"max time is {max(exe_time)}")
    print(f"median time is {statistics.median(exe_time)}")
    global MinexeTime
    global MaxexeTime
    global MedianTime
    MinexeTime += min(exe_time)
    MaxexeTime += max(exe_time)
    MedianTime += statistics.median(exe_time)


    time_5 = time.time()
    logits_tvm = m.get_output(0).numpy()
    #print("logits_tvm:", logits_tvm)
    time_6 = time.time()
    global mean_time
    mean_time += (time_6 - time_5)
    acc = evaluate(batch_graphs, logits_tvm)
    num = num + batch_size
    right_rel += acc

    return num,right_rel
def BVSM_B(DS):
    dataset = dgl.data.TUDataset(DS)
    # 2. 获取图的节点数并存储（图的索引，节点数）`
    graph_sizes = [(i, g[0].num_nodes()) for i, g in enumerate(dataset)]

    # 3. 根据节点数排序
    sorted_indices = [i for i, _ in sorted(graph_sizes, key=lambda x: x[1])]

    # 4. 创建一个新的数据集列表，按照排序后的索引
    dataset = [dataset[i] for i in sorted_indices]

    graph_sizes = [(i, g[0].num_nodes()) for i, g in enumerate(dataset)]
    sorted_indices = [i for i, _ in sorted(graph_sizes, key=lambda x: x[1])]
    sorted_dataset = [dataset[i] for i in sorted_indices]


    # 初始化第一个批次
    batch_graphs = [sorted_dataset[0]]
    prev_num_nodes = sorted_dataset[0][0].num_nodes()

    num = 0
    right_rel = 0
    for g in sorted_dataset[1:]:
        num_nodes = g[0].num_nodes()
        if num_nodes == prev_num_nodes:
            # 如果当前图的节点数与前一个相同，则添加到当前批次
            batch_graphs.append(g)
        else:
            # 节点数发生变化，处理当前批次
            addnum,addright_rel = process_batch(batch_graphs,sorted_dataset)
            num += addnum
            right_rel += addright_rel
            # 开始新的批次
            batch_graphs = [g]
            prev_num_nodes = num_nodes

    # 处理最后一个批次
    if batch_graphs:
        addnum,addright_rel = process_batch(batch_graphs,sorted_dataset)
        num += addnum
        right_rel += addright_rel

    print("num: ", num)
    acc = right_rel / num
    print("Test accuracy of TVM results: {:.2%}".format(acc))
    print("Min model inference total time: ", MinexeTime)
    print("Median model inference total time: ", MedianTime)
    print("Max model inference total time: ", MaxexeTime)









if __name__ == '__main__':
    # DAD_1_time = 0
    # DAD_2_time = 0

    MinexeTime = 0
    MaxexeTime = 0
    MedianTime = 0
    Graphconv1_time = 0
    Graphconv2_time = 0
    CreateW_time = 0
    TransformW_time = 0
    expendims_time = 0
    bias_var = 0
    biasAdd_time = 0
    AH_1_time = 0
    AH_2_time = 0
    HW_1_time = 0
    HW_2_time = 0
    activation_time = 0
    squeeze_time = 0
    mean_time = 0
    DS = str(sys.argv[1])
    BVSM_B(DS)