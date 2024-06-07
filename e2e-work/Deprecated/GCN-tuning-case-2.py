import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

# 1. 定义GCN模型
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return dgl.mean_nodes(g, 'h')

# 2. 生成随机图和节点特征作为示例输入
in_feats = 10
h_feats = 20 
num_classes = 5
num_nodes = 100
num_edges = 500

g = dgl.rand_graph(num_nodes, num_edges)
feat = torch.randn((num_nodes, in_feats))

# 3. 创建GCN模型实例
model = GCN(in_feats, h_feats, num_classes)

# 4. 将PyTorch模型转换为Relay计算图
input_shape = {'feat': feat.shape}
mod, params = relay.frontend.from_pytorch(model, input_shape)

# 5. 定义目标硬件(如CPU)
target = tvm.target.Target('llvm')

# 6. 使用Ansor提取任务并执行调优
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=200,
    runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
    measure_callbacks=[auto_scheduler.RecordToFile('gcn_tuning.log')],
)
tuner.tune(tune_option)

# 7. 编译调优后的模型
with auto_scheduler.ApplyHistoryBest('gcn_tuning.log'):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# 8. 在目标硬件上运行推理
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib['default'](dev))

# 将输入数据转换为NDArray
input_data = tvm.nd.array(feat.numpy())

# 设置输入
module.set_input('feat', input_data)

# 运行推理
module.run()

# 获取输出
output_data = module.get_output(0).numpy()

print(output_data)