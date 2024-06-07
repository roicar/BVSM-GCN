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


dataset = dgl.data.TUDataset("BZR")


dgl_g = dataset[0]
print("dgl_g: ",dgl_g)
print("dgl_g[0]: ",dgl_g[0])
num_layers = 1
num_hidden = 16
features = dgl_g[0].ndata['node_attr']
features = features.float()
infeat_dim = features.shape[1]
num_classes = dataset.num_classes
torch_model = GCN(in_feats = infeat_dim, h_feats = num_hidden, num_classes = num_classes)
optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
torch_model.load_state_dict(torch.load('BZR_model.pth'))
right_rel = 0.0
elapsed_Totaltime = 0
num = 0.0
from tvm import relay
from tvm.contrib import graph_executor
import tvm
from tvm import te

DAD_time = 0
mean_time = 0
Graphconv1_time = 0
Graphconv2_time = 0
CreateW_time = 0
expendims_time = 0
squeeze_time = 0
bias_time = 0
AH_1_time = 0
AH_2_time = 0
HW_1_time = 0
HW_2_time = 0
activation_time = 0

def GraphConv(layer_name, input_dim, output_dim, adj, input, norm=None, bias=True, activation=None):
    start1 = time.time()
    time_1 = time.time()
    if norm is not None:
        input = relay.multiply(input, norm)
    time_2 = time.time()
    global DAD_time
    DAD_time += (time_2 - time_1)

    start = time.time()
    weight = relay.var(layer_name + ".weight", shape=(1, input_dim, output_dim))
    end = time.time()
    global CreateW_time
    CreateW_time += (end - start)

    start = time.time()
    adj = relay.op.transform.expand_dims(adj, axis=0, num_newaxis=1)
    input = relay.op.transform.expand_dims(input, axis=0, num_newaxis=1)
    weight = relay.op.transform.expand_dims(weight, axis=0, num_newaxis=1)
    end = time.time()
    global expendims_time
    expendims_time += (end - start)

    start = time.time()
    dense_1 = relay.nn.batch_matmul(adj,input,transpose_b=False)  # 修改此行，使用稠密矩阵乘法
    end = time.time()
    if output_dim == 16:
        global AH_1_time
        AH_1_time += (end - start)
    else:
        global AH_2_time
        AH_2_time += (end - start)

    start = time.time()
    dense_2 = relay.nn.batch_matmul(dense_1, weight, transpose_b=False)
    end = time.time()
    if output_dim == 16:
        global HW_1_time
        HW_1_time += (end - start)
    else:
        global HW_2_time
        HW_2_time += (end - start)

    start = time.time()
    dense_2 = relay.op.transform.squeeze(dense_2,axis = 0)
    end = time.time()
    global squeeze_time
    squeeze_time += (end - start)

    

    if norm is not None:
        start = time.time()
        output = relay.multiply(dense_2, norm)
        end = time.time()
        #global DAD_time
        DAD_time += (end - start)

    if bias is True:
        start = time.time()
        _bias = relay.var(layer_name + ".bias", shape=(1, output_dim, 1))
        output = relay.nn.bias_add(output, _bias, axis=-1)
        end = time.time()
        global bias_time
        bias_time += (end - start)

    if activation is not None:
        start = time.time()
        output = activation(output)
        end = time.time()
        global activation_time
        activation_time += (end - start)
    end1 = time.time()
    if output_dim == 16:
        global Graphconv1_time
        Graphconv1_time += (end1 - start1)
    else:
        global Graphconv2_time
        Graphconv2_time += (end1 - start1)
    return output
    
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


    # Construct the 2-layer GCN
    layers = []
    layers.append(
        GraphConv(
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
        GraphConv(
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
    #print("shape:",output.shape())
    # Compile and run with TVM
    model_params = {}
    for param_tensor in torch_model.state_dict():
        model_params[param_tensor] = torch_model.state_dict()[param_tensor].numpy()

    for i in range(num_layers + 1):
        params["conv%d.weight" % (i+1)] = model_params["conv%d.weight" % (i+1)]
        params["conv%d.bias" % (i+1)] = model_params["conv%d.bias" % (i+1)]
    # Set the TVM build target
    target = "llvm"  # Currently only support `llvm` as target
    print("OK1")
    func = relay.Function(relay.analysis.free_vars(output), output)
    func = relay.build_module.bind_params_by_name(func, params)
    mod = tvm.IRModule()
    mod["main"] = func
    # Build with Relay
    print("OK2")
    with tvm.transform.PassContext(opt_level=4):
        lib = relay.build(mod, target, params=params)
    print("OK3")
    # Generate graph executor
    dev = tvm.device(target, 0)
    m = graph_executor.GraphModule(lib["default"](dev))
    print("OK4")

    start = time.time()
    m.run()
    end = time.time()
    elapsed_Totaltime += end - start

    with torch.no_grad():
        logits_torch = torch_model(in_feat = features, g = dgl_g[0])

    time_5 = time.time()
    logits_tvm = m.get_output(0).numpy()
    time_6 = time.time()
    mean_time += (time_6 - time_5)
    acc = evaluate(dgl_g, logits_tvm)
    num = num + 1.0
    if acc == 1:
        right_rel += 1.0
    #print("Print the first five outputs from TVM execution\n", logits_tvm[:5])


print("DAD_time is :", DAD_time)
print("CreateW time is :", CreateW_time)
print("expandims time is :", expendims_time)
print("AH_1 time is :", AH_1_time)
print("HW_1 time is :", HW_1_time)
print("AH_2 time is :", AH_2_time)
print("HW_2 time is :", HW_2_time)
print("bias time is :", bias_time)
print("squeeze time is :", squeeze_time)
print("activation time is :", activation_time)
print("the meanlogits time is :", mean_time)
print("GraphGonv1 time is :", Graphconv1_time)
print("GraphGonv2 time is :", Graphconv2_time)
print("num: ",num)

DAD_time = 0
mean_time = 0
Graphconv1_time = 0
Graphconv2_time = 0
CreateW_time = 0
expendims_time = 0
bias_time = 0
AH_1_time = 0
AH_2_time = 0
HW_1_time = 0
HW_2_time = 0
activation_time = 0
acc = right_rel/num
print("Test accuracy of TVM results: {:.2%}".format(acc))
print("model inference total time: ",elapsed_Totaltime)

