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

''' GCN --- TVM Tutorial version
class GCN(nn.Module):
    def __init__(self, g, n_infeat, n_hidden, n_classes, n_layers, activation):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(n_infeat, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_classes))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            # handle api changes for differnt DGL version
            if dgl.__version__ > "0.3":
                h = layer(self.g, h)
            else:
                h = layer(h, self.g)
        return h
'''

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


def evaluate(g, logits):
    label = g.ndata["node_labels"]
    test_mask = g.ndata["test_mask"]

    pred = logits.argmax(axis=1)
    acc = (torch.Tensor(pred[test_mask]) == label[test_mask]).float().mean()

    return acc


######################################################################
# Load the data and set up model parameters
# -----------------------------------------
"""
Parameters
----------
num_layer: int
    number of hidden layers

num_hidden: int
    number of the hidden units in the hidden layer

infeat_dim: int
    dimension of the input features

num_classes: int
    dimension of model output (Number of classes)
"""


dataset = dgl.data.TUDataset("DHFR")
dgl_g = dataset[0]
print("dgl_g: ",dgl_g)
print("dgl_g[0]: ",dgl_g[0])
num_layers = 1
num_hidden = 16
features = dgl_g[0].ndata['node_attr']
features = features.float()
infeat_dim = features.shape[1]
num_classes = dataset.num_classes
features = torch.FloatTensor(features)
#torch_model = GCN(dgl_g, infeat_dim, num_hidden, num_classes, num_layers, F.relu)
torch_model = GCN(in_feats = infeat_dim, h_feats = num_hidden, num_classes = num_classes)
optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)

# Here omit the training process
# Load the weights into the model
# model.pth is from DGL.py
torch_model.load_state_dict(torch.load('DHFR_model.pth'))


######################################################################
# Define Graph Convolution Layer in Relay

from tvm import relay
from tvm.contrib import graph_executor
import tvm
from tvm import te


def GraphConv(layer_name, input_dim, output_dim, adj, input, dMM_time,  norm=None, bias=True, activation=None):
    """
    Parameters
    ----------
    layer_name: str
    Name of layer

    input_dim: int
    Input dimension per node feature

    output_dim: int,
    Output dimension per node feature

    adj: namedtuple,
    Graph representation (Adjacency Matrix) in Sparse Format (`data`, `indices`, `indptr`),
    where `data` has shape [num_nonzeros], indices` has shape [num_nonzeros], `indptr` has shape [num_nodes + 1]

    input: relay.Expr,
    Input feature to current layer with shape [num_nodes, input_dim]

    norm: relay.Expr,
    Norm passed to this layer to normalize features before and after Convolution.

    bias: bool
    Set bias to True to add bias when doing GCN layer

    activation: <function relay.op.nn>,
    Activation function applies to the output. e.g. relay.nn.{relu, sigmoid, log_softmax, softmax, leaky_relu}

    Returns
    ----------
    output: tvm.relay.Expr
    The Output Tensor for this layer [num_nodes, output_dim]
    """
    if norm is not None:
        input = relay.multiply(input, norm)

    weight = relay.var(layer_name + ".weight", shape=(input_dim, output_dim))
    weight_t = relay.transpose(weight)
    dense = relay.nn.dense(weight_t, input)
    start = time.time()
    output = relay.nn.sparse_dense(dense, adj)
    end = time.time()
    dMM_time = dMM_time + (end - start)
    print("one SparseMM operator time is:",(end - start))
    output_t = relay.transpose(output)
    if norm is not None:
        output_t = relay.multiply(output_t, norm)
    if bias is True:
        _bias = relay.var(layer_name + ".bias", shape=(output_dim, 1))
        output_t = relay.nn.bias_add(output_t, _bias, axis=-1)
    if activation is not None:
        output_t = activation(output_t)
    return output_t


######################################################################
# Prepare the parameters needed in the GraphConv layers




def prepare_params(g):
    params = {}
    params["infeats"] = g.ndata["node_attr"].numpy().astype("float32")

    # Generate adjacency matrix
    nx_graph = dgl.to_networkx(g)
    adjacency = nx.to_scipy_sparse_array(nx_graph)
    params["g_data"] = adjacency.data.astype("float32")
    params["indices"] = adjacency.indices.astype("int32")
    params["indptr"] = adjacency.indptr.astype("int32")

    # Normalization w.r.t. node degrees
    degs = [g.in_degrees(i) for i in range(g.number_of_nodes())]
    params["norm"] = np.power(degs, -0.5).astype("float32")
    params["norm"] = params["norm"].reshape((params["norm"].shape[0], 1))

    return params


params = prepare_params(dgl_g[0])

# Check shape of features and the validity of adjacency matrix
assert len(params["infeats"].shape) == 2
assert (
    params["g_data"] is not None and params["indices"] is not None and params["indptr"] is not None
)
assert params["infeats"].shape[0] == params["indptr"].shape[0] - 1

######################################################################
# Put layers together
# -------------------

# Define input features, norms, adjacency matrix in Relay
infeats = relay.var("infeats", shape=features.shape)
norm = relay.Constant(tvm.nd.array(params["norm"]))
g_data = relay.Constant(tvm.nd.array(params["g_data"]))
indices = relay.Constant(tvm.nd.array(params["indices"]))
indptr = relay.Constant(tvm.nd.array(params["indptr"]))

Adjacency = namedtuple("Adjacency", ["data", "indices", "indptr"])
adj = Adjacency(g_data, indices, indptr)

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
        dMM_time=dMM_time,
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
        dMM_time=dMM_time,
        norm=norm,
        activation=None,
    )
)

# Analyze free variables and generate Relay function
output = layers[-1]

######################################################################
# Compile and run with TVM
# ------------------------
#
# Export the weights from PyTorch model to Python Dict
model_params = {}
for param_tensor in torch_model.state_dict():
    model_params[param_tensor] = torch_model.state_dict()[param_tensor].numpy()

'''
for i in range(num_layers + 1):
    params["layers.%d.weight" % (i)] = model_params["layers.%d.weight" % (i)]
    params["layers.%d.bias" % (i)] = model_params["layers.%d.bias" % (i)]
'''
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
with tvm.transform.PassContext(opt_level=0):  # Currently only support opt_level=0
    lib = relay.build(mod, target, params=params)

# Generate graph executor
dev = tvm.device(target, 0)
m = graph_executor.GraphModule(lib["default"](dev))

######################################################################
# Run the TVM model, test for accuracy and verify with DGL
# --------------------------------------------------------

start = time.time()
m.run()
end = time.time()
elapsed_time_3 = end - start
print("TVM first Graph Inference_time:",elapsed_time_3)
total_time = 0
# 定义一个函数来执行单个图的推理
def infer_single_graph(graph,total_time):
    # 设置输入特征
    infeats = graph.ndata['node_attr'].float()
    params = prepare_params(dgl_g[0])

    # Check shape of features and the validity of adjacency matrix
    assert len(params["infeats"].shape) == 2
    assert (
        params["g_data"] is not None and params["indices"] is not None and params["indptr"] is not None
    )
    assert params["infeats"].shape[0] == params["indptr"].shape[0] - 1
    
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
    with tvm.transform.PassContext(opt_level=0):  # Currently only support opt_level=0
        lib = relay.build(mod, target, params=params)

    # Generate graph executor
    dev = tvm.device(target, 0)
    m = graph_executor.GraphModule(lib["default"](dev))
    
    # 执行推理
    start = time.time()
    m.run()
    end = time.time()
    
    total_time = total_time + end - start
    # 获取输出
    logits = m.get_output(0).asnumpy()
    return logits,total_time


from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=5, drop_last=False
)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=5, drop_last=False
)


it = iter(test_dataloader)
batch = next(it)
print(batch)


batched_graph, labels = batch

# Recover the original graph elements from the minibatch
graphs = dgl.unbatch(batched_graph)
print("The original graphs in the minibatch:")
print(graphs)

num_correct = 0
num_tests = 0
elapsed_time = 0
for batched_graph, labels in test_dataloader:
    start = time.time()
    pred = torch_model(batched_graph, batched_graph.ndata["node_attr"].float())
    end = time.time()
    elapsed_time = elapsed_time + (end - start)
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)
    print("pred: ",pred)
print("num_correct: ",num_correct)
print("num_tests: ",num_tests)
print("DGL Test accuracy:", num_correct / num_tests)
print("DGL Infer time:", elapsed_time)

# 遍历所有test图并进行推理
all_logits = []

for batched_graph, labels in test_dataloader:
    graphs = dgl.unbatch(batched_graph)
    for g in graphs:
        print("current total_time:",total_time)
        logits,total_time = infer_single_graph(g,total_time)
        all_logits.append(logits)
        print("inner")
    print("outer")

print("all_logits: ",all_logits)
print("Total TVM inference time for all graphs:", total_time)

print("dgl_g",dgl_g)
start = time.time()
with torch.no_grad():
    logits_torch = torch_model(in_feat = features, g = dgl_g[0])
end = time.time()
elapsed_time_2 = end - start
print("Pytorch first layer Inference_time:",elapsed_time_2)
logits_tvm = m.get_output(0).numpy()
print("Print the first five outputs from TVM execution\n", logits_tvm[:5])



'''
acc = evaluate(dgl_g, logits_torch.numpy())
print("Test accuracy of DGL results: {:.2%}".format(acc))
print("Print the first five outputs from DGL execution\n", logits_torch)


acc = evaluate(dgl_g[0], logits_tvm)
print("Test accuracy of TVM results: {:.2%}".format(acc))

import tvm.testing

# Verify the results with the DGL model
tvm.testing.assert_allclose(logits_torch, logits_tvm, atol=1e-3)
'''






