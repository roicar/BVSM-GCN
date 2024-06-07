import tvm
from tvm import relay
import numpy as np

# 定义图卷积层
def gcn_layer(features, weight, adj, num_nodes):
    # 以下代码假定adj是稀疏表示的
    # 如果adj是稠密的，你可以直接进行矩阵乘法
    print("adj: ",adj)
    #print("adj.input_dim: ",adj.input_dim)
    print("num_nodes: ",num_nodes)
    #print("adj.type.shape: ",adj.struct_info)
    adj = relay.nn.dense(adj, relay.const(np.identity(num_nodes), dtype="float32"))
    features = relay.nn.dense(features, weight)
    out = relay.nn.dense(adj, features)
    return out

# 构建GCN模型
def build_gcn_model(num_nodes, num_features, num_classes):
    # 输入特征和邻接矩阵
    features = relay.var("features", shape=(num_nodes, num_features))
    adj = relay.var("adj", shape=(num_nodes, num_nodes))
    
    # GCN层
    weight1 = relay.var("weight1", shape=(num_features, num_classes))
    out1 = gcn_layer(features, weight1, adj,num_nodes)
    
    # 可以添加更多层
    
    # 创建函数
    func = relay.Function([features, adj, weight1], out1)
    
    return func

# 构建模型
num_nodes = 34  # 示例
num_features = 7  # 示例
num_classes = 2  # 示例
model = build_gcn_model(num_nodes, num_features, num_classes)

# 优化
with relay.build_config(opt_level=2):
    graph, lib, params = relay.build(model, target="llvm")

# 这里可以输出graph, lib, params进行进一步的处理和运行



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























