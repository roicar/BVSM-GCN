"""
Building a Graph Convolutional Network combined with BVSM
======================================
**Author**: Daiwen

Tasks:
1.construct a simple GNN(GCN) as same as that in PyG/DGL.
2.load parameters into model

"""
import tvm
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
from tvm import te, auto_scheduler
import glob
import os

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


dataset = dgl.data.TUDataset("COX2")
print("dataset[0]:", dataset[0])
#重新按照节点数排序
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
torch_model.load_state_dict(torch.load('COX2_model.pth'))
right_rel = 0.0
elapsed_Totaltime = 0
num = 0.0
from tvm import relay
from tvm.contrib import graph_executor
import tvm
from tvm import te

DAD_time = 0
KeyMM_time = 0
denseMM_time = 0
mean_time = 0
Graphconv_time = 0
class GraphConv(nn.Module):
    def __init__(self, layer_name, input_dim, output_dim, adj, input, norm=None, bias=True, activation=None):
        super(GraphConv, self).__init__()
        self.layer_name = layer_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj = adj
        self.input = input
        self.norm = norm
        self.bias = bias
        self.activation = activation

    def forward(self):
        start1 = time.time()
        if self.norm is not None:
            self.input = relay.multiply(self.input, self.norm)
        weight = relay.var(self.layer_name + ".weight", shape=(self.input_dim, self.output_dim))
        weight_t = relay.transpose(weight)
        dense = relay.nn.dense(weight_t, self.input)
        output = relay.op.nn.dense(dense, self.adj)
        output_t = relay.transpose(output)
        if self.norm is not None:
            output_t = relay.multiply(output_t, self.norm)
        if self.bias:
            _bias = relay.var(self.layer_name + ".bias", shape=(self.output_dim, 1))
            output_t = relay.nn.bias_add(output_t, _bias, axis=-1)
        if self.activation is not None:
            output_t = self.activation(output_t)
        return output_t



# 在进行端到端构建前，独立对GraphConv中的矩阵乘法进行调优
# 这需要你提供具体的输入尺寸和数据类型
# 示例调优代码（假设任务已经定义）
from tvm.auto_scheduler import SearchTask,TuningOptions, ApplyHistoryBest


def tune_graph_conv(features_shape, adj_shape, num_hidden, log_file, target="llvm"):
    # 定义输入张量的形状
    features_shape = (16, adj_shape[1])  # 特征矩阵的形状，例如 (num_nodes, in_features)
    # weight_shape = (features_shape[1],adj_shape[1])  # 假设为简化的全连接层权重形状

    # 创建一个简单的矩阵乘法模型用于调优
    W = relay.var("W", shape=adj_shape, dtype="float32")
    A = relay.var("A", shape=features_shape, dtype="float32")

    dense = relay.op.nn.dense(A, W)
    func = relay.Function([A, W], dense)
    print("func:",func)
    # Create a tuning task
    tasks = auto_scheduler.extract_tasks(func, {}, target)

    # 这里只取第一个任务来调优
    task = tasks[0]
    # task = SearchTask(func, {})

    # Set tuning options
    tune_options = TuningOptions(
        num_measure_trials=200,
        runner=auto_scheduler.LocalRunner(repeat=5, timeout=20, min_repeat_ms=100),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        early_stopping=200
    )

    # Execute tuning
    #tuner = auto_scheduler.TaskScheduler(task, task_weights=[1] * len(task))
    tuner = auto_scheduler.TaskScheduler(task, task_weights=[1] * len(task))
    tuner.tune(tune_options)





graph = 1      
for dgl_g in dataset:
    print("The {} graph has been processed".format(graph))
    graph += 1
    features = dgl_g[0].ndata['node_attr'].float()
    infeat_dim = features.shape[1]
    num_nodes = dgl_g[0].number_of_nodes()

    # 调整为需要的形状
    features_shape = (num_nodes, infeat_dim)
    adj_shape = (num_nodes, num_nodes)  # 假设邻接矩阵是方阵

    # 为当前图执行调优
    log_file = "tune_COX2_"+str(adj_shape[1])+"_"+str(features_shape[1])+".json"
    tune_graph_conv(features_shape, adj_shape, num_hidden, log_file)

    features = torch.FloatTensor(features)
    params = prepare_params(dgl_g[0])
    # Check shape of features and the validity of adjacency matrix
    assert len(params["infeats"].shape) == 2
    assert (
            params["adj"] is not None
    )
    assert params["infeats"].shape[0] == params["adj"].shape[0]

    # Ansor Autotuning

    # Put layers together
    infeats = relay.var("infeats", shape=params["infeats"].shape)
    norm = relay.Constant(tvm.nd.array(params["norm"]))
    adj = relay.const(tvm.nd.array(params["adj"]))
    # print("adj_Today:",tvm.nd.array(params["adj"]))
    # Construct the 2-layer GCN
    layers = []
    layers.append(
        GraphConv(
            # layer_name="layers.0",
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
            # layer_name="layers.1",
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
        params["conv%d.weight" % (i + 1)] = model_params["conv%d.weight" % (i + 1)]
        params["conv%d.bias" % (i + 1)] = model_params["conv%d.bias" % (i + 1)]
    # Set the TVM build target
    target = "llvm"  # Currently only support `llvm` as target

    func = relay.Function(relay.analysis.free_vars(output), output)
    func = relay.build_module.bind_params_by_name(func, params)
    mod = tvm.IRModule()
    mod["main"] = func
    # 从调优结果中加载最佳配置并构建模型
    with ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target="llvm", params=params)
            # Generate graph executor
            dev = tvm.device(target, 0)
            # dev = tvm.cuda()
            m = graph_executor.GraphModule(lib["default"](dev))

            start = time.time()
            m.run()
            end = time.time()
            elapsed_Totaltime += end - start

            time_5 = time.time()
            logits_tvm = m.get_output(0).numpy()
            time_6 = time.time()
            mean_time += (time_6 - time_5)
            acc = evaluate(dgl_g, logits_tvm)
            num = num + 1.0
            if acc == 1:
                right_rel += 1.0
            # print("Print the first five outputs from TVM execution\n", logits_tvm[:5])






print("GraphGonv time is :", Graphconv_time)
print("DAD_time is :", DAD_time)
print("the denseMM time is :", denseMM_time)
print("the KeyMM time is :", KeyMM_time)
print("the meanlogits time is :", mean_time)
print("num: ",num)


acc = right_rel/num
print("Test accuracy of TVM results: {:.2%}".format(acc))
print("model inference total time: ",elapsed_Totaltime)

