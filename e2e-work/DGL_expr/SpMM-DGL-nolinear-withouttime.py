"""
Here we can infer GCN with DGL+Pytorch
We keep the train module in the front

"""

import os

import numpy as np

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import timeit
import sys

def preprocess_graphs(graphs):
    processed_graphs = []
    for g in graphs:
        #print("g:",g)
        t = g[1]
        g = g[0]
        # add selfloop
        #print("g:",g)
        g = dgl.add_self_loop(g)

        # normalization
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs,-0.5)
        #norm = norm.to(g.device).unsqueeze(1)
        #print(g)
        # norm = torch.diag(norm)
        g.ndata['norm'] = norm
        processed_graphs.append((g,t))
    return processed_graphs
def SpMM(DS, in_feats):

    #Notice:You need to change parameter of infeat_nums after changing dataset
    #dataset = dgl.data.TUDataset("AIDS")
    #dataset = dgl.data.TUDataset("BZR")
    #dataset = dgl.data.TUDataset("COX2")
    #dataset = dgl.data.TUDataset("DHFR")
    dataset = dgl.data.TUDataset(DS)
    # 1. 读取数据集

    # 2. 获取图的节点数并存储（图的索引，节点数）
    graph_sizes = [(i, g[0].num_nodes()) for i, g in enumerate(dataset)]

    # 3. 根据节点数排序
    sorted_indices = [i for i, _ in sorted(graph_sizes, key=lambda x: x[1])]

    # 4. 创建一个新的数据集列表，按照排序后的索引
    sorted_dataset = [dataset[i] for i in sorted_indices]

    subset = int(len(sorted_dataset))
    inference_subset = sorted_dataset[:subset]



    dataset = preprocess_graphs(inference_subset)




    from dgl.dataloading import GraphDataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    #from torch.utils.data.sampler import SequentialSampler






    from torch.utils.data.sampler import SequentialSampler
    num_examples = len(dataset)
    #num_train = int(num_examples * 0.01)
    num_test = int(num_examples )

    #train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SequentialSampler(torch.arange(num_test))

    #train_dataloader = GraphDataLoader(
    #    dataset, sampler=train_sampler, batch_size=5, drop_last=False
    #)

    # Here we compare with BVSM-M, batch_size=num_test
    start = time.time()

    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=32, drop_last=False
    )
    end = time.time()
    print("The Dataload time is : ", (end - start))

    it = iter(test_dataloader)
    batch = next(it)
    print(batch)
    batched_graph, labels = batch
    print(
        "Number of nodes for each graph element in the batch:",
        batched_graph.batch_num_nodes(),
    )
    print(
        "Number of edges for each graph element in the batch:",
        batched_graph.batch_num_edges(),
    )

    # Recover the original graph elements from the minibatch
    graphs = dgl.unbatch(batched_graph)
    print("The original graphs in the minibatch:")
    #print(graphs)

    # Initialize the model
    model = SparseGCN(in_feats, 16, 2)



    # Create the model with given dimensions
    #print("OK7")
    #model = DenseGCN(3, 16, dataset.num_classes)
    #print("OK8")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    ##########################################################################
    # Training Loop
    # Skip it when you only need inference



    #state_dict = torch.load('BZR_model_Dense.pth')
    state_dict = torch.load('../model_parameters/'+DS+'_model.pth')



    model.load_state_dict(state_dict)

    model.eval()

    num_correct = 0
    num_tests = 0
    elapsed_time = 0
    test_num = 0
    for batched_graph, labels in test_dataloader:
        print("test_num: ",test_num)
        test_num += 1
        #print("batched_graph: ",batched_graph)
        length = len(labels)
        print("-----------------------------------------------------")
        #print("labels: ",labels)
        #print("labels.squeeze(): ",labels.squeeze())
        labels = labels.squeeze()
        start = time.time()
        pred = model(batched_graph, batched_graph.ndata["node_attr"].float())
        end = time.time()
        elapsed_time = elapsed_time + (end - start)

        print("pred: ",pred)
        print("pred.argmax(1): ",pred.argmax(1))
        print("pred.argmax(1).squeeze(): ",pred.argmax(1).squeeze())
        print("(pred.argmax(1)==labels).sum: ",(pred.argmax(1) == labels).sum())
        print("(pred.argmax(1)==labels).sum.item(): ",(pred.argmax(1) == labels).sum().item())


        print("num_correct_before:",num_correct)
        num_correct += (pred.argmax(1) == labels).sum().item()
        print("num_correct_after:",num_correct)
        num_tests += length

    print("---------------------------------------------------------")
    print("num_correct: ",num_correct)
    print("num_tests: ",num_tests)
    print("Test accuracy:", num_correct / num_tests)
    print("Total_time:", elapsed_time)
    print("---------------------------------------------------------")
    print("init_1_time: ", init_1_time)
    print("init_2_time: ", init_2_time)
    print("KeyMM_1_time: ", KeyMM_1_time)
    print("KeyMM_2_time: ", KeyMM_2_time)
    print("linear_1_time: ", linear_1_time)
    print("linear_2_time: ", linear_2_time)
    print("bias_1_time: ", bias_1_time)
    print("bias_2_time: ", bias_2_time)
    print("numpy_transform_1_A_time: ", numpy_transform_1_B_time)
    print("numpy_transform_2_A_time: ", numpy_transform_2_B_time)
    print("numpy_transform_1_B_time: ", numpy_transform_1_W_time)
    print("numpy_transform_2_B_time: ", numpy_transform_2_W_time)
    print("numpy_transform_1_bias_time: ", numpy_transform_1_bias_time)
    print("numpy_transform_2_bias_time: ", numpy_transform_2_bias_time)
    print("torch_transform_1_time: ", torch_transform_1_time)
    print("torch_transform_2_time: ", torch_transform_2_time)

    print("---------------------------------------------------------")
    print("get_adj_from_graph_time: ", get_adj_from_graph)
    print("get_norm_time: ", get_norm)
    print("norm_transform_time: ", norm_transform)
    print("DAD_time: ", DAD_time)
    print("---------------------------------------------------------")
    print("Normalization_time: ",Normalization)
    print("conv1_time: ",conv1)
    print("relu_time: ",relu)
    print("conv2_time: ",conv2)
    print("meanNode_time: ",meanNode_time)
    print("---------------------------------------------------------")
    print("Infer time:", DAD_time + conv1 + conv2 + relu)






class SparseGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        start = time.time()
        super(SparseGCNConv, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_feats, out_feats))
        self.bias = nn.Parameter(torch.randn(out_feats))
        self.infeats = in_feats
        self.out_feats = out_feats
        end = time.time()
        if self.out_feats == 16:
            global init_1_time
            init_1_time += (end - start)
        else:
            global init_2_time
            init_2_time += (end - start)
    def forward(self, adj, features):

        sparse_feat = torch.sparse.mm(adj, features)
        sparse_feat = sparse_feat.to(torch.float).detach().numpy()  # 确保转换为numpy数组

        weight = self.weight.detach().numpy()  # 将权重转换为numpy数组
        dense_feat = np.dot(sparse_feat, weight)



        if self.bias is not None:
            bias = self.bias.detach().numpy() #detach是方便Tensor调用numpy()
            dense_feat += bias


        dense_feat = torch.from_numpy(dense_feat)

        return dense_feat
    
    
class SparseGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SparseGCN, self).__init__()
        self.conv1 = SparseGCNConv(in_feats, h_feats)
        self.conv2 = SparseGCNConv(h_feats, num_classes)
        
        
    def normalize_adjacency(self, g):
        start  =time.time()
        adj = g.adjacency_matrix()
        end = time.time()
        global get_adj_from_graph
        get_adj_from_graph += (end - start)


        start = time.time()
        norm = g.ndata['norm']
        end = time.time()
        global get_norm
        get_norm += (end - start)


        start = time.time()
        norm = torch.diag(norm).to_sparse_coo()
        end = time.time()
        global norm_transform
        norm_transform += (end - start)


        time_3 = time.time()
        adj_normalized = torch.sparse.mm(norm, torch.sparse.mm(adj, norm))
        time_4 = time.time()
        global DAD_time
        DAD_time += (time_4 - time_3)

        return adj_normalized


    def forward(self, g, features):
        start = time.time()
        adj_normalized = self.normalize_adjacency(g)
        end = time.time()
        global Normalization
        Normalization += (end - start)

        start = time.time()
        x = self.conv1(adj_normalized, features)
        end = time.time()
        global conv1
        conv1 += (end - start)

        start = time.time()
        x = F.relu(x)
        end = time.time()
        global relu
        relu += (end - start)

        start = time.time()
        x = self.conv2(adj_normalized, x)
        end = time.time()
        global conv2
        conv2 += (end - start)


        
        start = time.time()
        g.ndata['h'] = x
        meannode = dgl.mean_nodes(g, 'h')

        end = time.time()
        global meanNode_time
        meanNode_time += (end-start)
        
        return meannode




if __name__ == '__main__':
    DAD_time = 0
    init_1_time = 0
    init_2_time = 0
    KeyMM_1_time = 0
    KeyMM_2_time = 0
    linear_1_time = 0
    linear_2_time = 0
    bias_1_time = 0
    bias_2_time = 0
    numpy_transform_1_B_time = 0
    numpy_transform_2_B_time = 0
    numpy_transform_1_W_time = 0
    numpy_transform_2_W_time = 0
    numpy_transform_1_bias_time = 0
    numpy_transform_2_bias_time = 0
    torch_transform_1_time = 0
    torch_transform_2_time = 0
    meanNode_time = 0

    get_adj_from_graph = 0
    adj_todense = 0
    get_norm = 0
    norm_transform = 0
    DAD_time1 = 0
    DAD_time2 = 0
    DAD_time3 = 0
    DAD_time4 = 0

    Normalization = 0
    conv1 = 0
    relu = 0
    conv2 = 0
    DS = str(sys.argv[1])
    in_feats = int(sys.argv[2])
    SpMM(DS, in_feats)


