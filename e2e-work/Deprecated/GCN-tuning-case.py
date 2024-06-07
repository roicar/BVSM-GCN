import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import torch
import dgl
import numpy as np
from dgl.nn.pytorch import GraphConv

# 1. 定义模型
class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return dgl.mean_nodes(g, "h")

# 2. 准备模型和数据
model = GCN(10, 16, 5)  # 示例参数
dummy_input = dgl.rand_graph(10, 20)  # 随机图
dummy_features = torch.rand((10, 10))  # 节点特征

# 3. 将模型转换为Relay计算图
shape_dict = {"input": dummy_features.shape}
mod, params = relay.frontend.from_pytorch(model, shape_dict)
target = "llvm"

# 4. 提取任务并进行调优
tasks = auto_scheduler.extract_tasks(mod["main"], params, target)
tuner = auto_scheduler.TaskScheduler(tasks)
tune_options = auto_scheduler.TuningOptions(
    num_measure_trials=200,  # 调优试验次数
    runner=auto_scheduler.LocalRunner(repeat=3, enable_cpu_cache_flush=True),
    measure_callbacks=[auto_scheduler.RecordToFile("gcn_tuning.json")]
)
tuner.tune(tune_options)

# 5. 使用调优日志编译模型
with auto_scheduler.ApplyHistoryBest("gcn_tuning.json"):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target, params=params)

# 6. 运行推理
dev = tvm.device(target, 0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("input", tvm.nd.array(dummy_features.numpy()))
module.run()
output = module.get_output(0).numpy()

print(output)