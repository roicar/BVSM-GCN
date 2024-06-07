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
from collections import namedtuple
from dgl.nn.pytorch import GraphConv
from tvm.contrib.download import download_testdata
import numpy as np
import networkx as nx
from tvm import relay
from tvm.contrib import graph_executor
import tvm
import sys
import statistics


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


import torch
import numpy as np

def evaluate(g, logits):
    
    label = g[1]
    g = g[0]
    
    pred = logits.mean(0).argmax().item()
    print("pred: ",pred)
    print("label: ",label.item())
    if pred == label.item():
        acc = 1
    else:
        acc = 0

    return acc  




import numpy as np
import networkx as nx

def prepare_params(g):
    params = {}
    params["infeats"] = g.ndata["node_attr"].numpy().astype("float32")

    # Generate adjacency matrix in dense format
    nx_graph = dgl.to_networkx(g)
    adjacency = nx.to_scipy_sparse_array(nx_graph).toarray()
    #print("adjacency:",adjacency)
    # 创建对角单位矩阵
    diagonal_unit_matrix = np.eye(adjacency.shape[0], dtype=np.float32)
    params["adj"] = adjacency.astype("float32")+ diagonal_unit_matrix #添加邻接矩阵
    #params["indices"] = adjacency.indices.astype("int32")
    #params["indptr"] = adjacency.indptr.astype("int32")

    # Normalization w.r.t. node degrees
    degs = [g.in_degrees(i) for i in range(g.number_of_nodes())]
    params["norm"] = np.power(degs, -0.5).astype("float32")
    params["norm"] = params["norm"].reshape((params["norm"].shape[0], 1))

    return params


def Graphconv(layer_name, input_dim, output_dim, adj, input, norm=None, bias=True, activation=None):
    start1 = time.time()

    if norm is not None:
        time_1 = time.time()
        input = relay.multiply(input, norm)
        time_2 = time.time()
        if output_dim == 16:
            global DAD_1_1_time
            DAD_1_1_time += (time_2 - time_1)
        else:
            global DAD_2_1_time
            DAD_2_1_time += (time_2 - time_1)

    weight = relay.var(layer_name + ".weight", shape=(input_dim, output_dim))
    weight_t = relay.transpose(weight)
    time_3 = time.time()
    dense = relay.nn.dense(weight_t, input)  # 修改此行，使用稠密矩阵乘法
    time_4 = time.time()
    global denseMM_time
    denseMM_time += (time_4 - time_3)

    # Only record the first layer time

    start = time.time()
    output = relay.nn.dense(dense, adj)
    end = time.time()
    global KeyMM_time
    KeyMM_time = KeyMM_time + (end - start)
    print("one denseMM operator time is:", (end - start))

    output_t = relay.transpose(output)
    if norm is not None:
        time_1 = time.time()
        output_t = relay.multiply(output_t, norm)
        time_2 = time.time()
        if output_dim == 16:
            global DAD_1_2_time
            DAD_1_2_time += (time_2 - time_1)
        else:
            global DAD_2_2_time
            DAD_2_2_time += (time_2 - time_1)
    if bias is True:
        _bias = relay.var(layer_name + ".bias", shape=(output_dim, 1))
        output_t = relay.nn.bias_add(output_t, _bias, axis=-1)
    if activation is not None:
        start = time.time()
        output_t = activation(output_t)
        end = time.time()
        global relu
        relu += (end - start)
    end1 = time.time()
    if output_dim == 16:
        global Graphconv_layer1_time
        Graphconv_layer1_time += (end1 - start1)
    else:
        global Graphconv_layer2_time
        Graphconv_layer2_time += (end1 - start1)
    return output_t



def tvm_dense(DS):
    dataset = dgl.data.TUDataset(DS)
    # 2. 获取图的节点数并存储（图的索引，节点数）
    graph_sizes = [(i, g[0].num_nodes()) for i, g in enumerate(dataset)]

    # 3. 根据节点数排序
    sorted_indices = [i for i, _ in sorted(graph_sizes, key=lambda x: x[1])]

    # 4. 创建一个新的数据集列表，按照排序后的索引
    dataset = [dataset[i] for i in sorted_indices]

    dgl_g = dataset[0]
    print("dgl_g: ",dgl_g)
    print("dgl_g[0]: ",dgl_g[0])
    num_layers = 1
    num_hidden = 16
    features = dgl_g[0].ndata['node_attr']
    features = features.float()
    infeat_dim = features.shape[1]
    num_classes = 2
    torch_model = GCN(in_feats = infeat_dim, h_feats = num_hidden, num_classes = num_classes)
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
    torch_model.load_state_dict(torch.load(DS+'_model.pth'))
    right_rel = 0.0
    elapsed_Totaltime = 0
    num = 0.0

    MinexeTime = 0
    MaxexeTime = 0
    MedianTime = 0
    graph = 1
    for dgl_g in dataset:
        print("The {} graph has been processed".format(graph))
        graph += 1
        features = dgl_g[0].ndata['node_attr']
        features = features.float()
        infeat_dim = features.shape[1]
        features = torch.FloatTensor(features)
        params = prepare_params(dgl_g[0])
        # Check shape of features and the validity of adjacency matrix
        assert len(params["infeats"].shape) == 2
        assert (
            params["adj"] is not None
        )
        assert params["infeats"].shape[0] == params["adj"].shape[0]

        # Put layers together
        infeats = relay.var("infeats", shape=params["infeats"].shape)
        norm = relay.Constant(tvm.nd.array(params["norm"]))
        adj = relay.const(tvm.nd.array(params["adj"]))
        # print("adj_Today:",tvm.nd.array(params["adj"]))
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
                norm=norm,
                activation=relay.nn.relu,
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
                norm=norm,
                activation=None,
            )
        )

        # Analyze free variables and generate Relay function
        output = layers[-1]
        # Compile and run with TVM
        model_params = {}
        for param_tensor in torch_model.state_dict():
            model_params[param_tensor] = torch_model.state_dict()[param_tensor].numpy()

        for i in range(num_layers + 1):
            params["conv%d.weight" % (i+1)] = model_params["conv%d.weight" % (i+1)]
            params["conv%d.bias" % (i+1)] = model_params["conv%d.bias" % (i+1)]
        # Set the TVM build target
        target = "llvm"  # Currently only support `llvm` as target

        func = relay.Function(relay.analysis.free_vars(output), output)
        func = relay.build_module.bind_params_by_name(func, params)
        mod = tvm.IRModule()
        mod["main"] = func
        # Build with Relay
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target, params=params)

        # Generate graph executor
        dev = tvm.device(target, 0)
        m = graph_executor.GraphModule(lib["default"](dev))

        exe_time = []
        for _ in range(1000):
            start = time.time()
            m.run()
            end = time.time()
            exe_time.append(end - start)

        print(f"min time is {min(exe_time)}")
        print(f"max time is {max(exe_time)}")
        print(f"median time is {statistics.median(exe_time)}")

        MinexeTime += min(exe_time)
        MaxexeTime += max(exe_time)
        MedianTime += statistics.median(exe_time)

        time_5 = time.time()
        logits_tvm = m.get_output(0).numpy()
        time_6 = time.time()
        global mean_time
        mean_time += (time_6 - time_5)
        acc = evaluate(dgl_g, logits_tvm)
        num = num + 1.0
        if acc == 1:
            right_rel += 1.0
        #print("Print the first five outputs from TVM execution\n", logits_tvm[:5])

    #print("GraphGonv time is :", Graphconv_time)
    #print("DAD_time is :", DAD_time)
    print("the denseMM time is :", denseMM_time)
    print("the sparseMM time is :", KeyMM_time)
    print("the meanlogits time is :", mean_time)
    print("num: ",num)


    acc = right_rel/num
    print("Test accuracy of TVM results: {:.2%}".format(acc))
    print("Min model inference total time: ", MinexeTime)
    print("Median model inference total time: ", MedianTime)
    print("Max model inference total time: ", MaxexeTime)

if __name__ == '__main__':
    DAD_1_1_time = 0
    DAD_1_2_time = 0
    DAD_2_1_time = 0
    DAD_2_2_time = 0
    denseMM_time = 0
    KeyMM_time = 0
    relu = 0
    Graphconv_layer1_time = 0
    Graphconv_layer2_time = 0
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
    tvm_dense(DS)