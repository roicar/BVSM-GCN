"""
Training a GNN for Graph Classification
=======================================

By the end of this tutorial, you will be able to

-  Load a DGL-provided graph classification dataset.
-  Understand what *readout* function does.
-  Understand how to create and use a minibatch of graphs.
-  Build a GNN-based graph classification model.
-  Train and evaluate the model on a DGL-provided dataset.

(Time estimate: 18 minutes)
"""

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import timeit
######################################################################
# Overview of Graph Classification with GNN
# -----------------------------------------
#
# Graph classification or regression requires a model to predict certain
# graph-level properties of a single graph given its node and edge
# features.  Molecular property prediction is one particular application.
#
# This tutorial shows how to train a graph classification model for a
# small dataset from the paper `How Powerful Are Graph Neural
# Networks <https://arxiv.org/abs/1810.00826>`__.
#
# Loading Data
# ------------
#

#Notice:You need to change parameter of infeat_nums after changing dataset
#dataset = dgl.data.TUDataset("AIDS")
#dataset = dgl.data.TUDataset("BZR")
#dataset = dgl.data.TUDataset("COX2")
dataset = dgl.data.TUDataset("DHFR")

######################################################################
# The dataset is a set of graphs, each with node features and a single
# label. One can see the node feature dimensionality and the number of
# possible graph categories of ``GINDataset`` objects in ``dim_nfeats``
# and ``gclasses`` attributes.
#

print("Max number of node:", dataset.max_num_node)
print("Number of graph categories:", dataset.num_classes)


from dgl.dataloading import GraphDataLoader

######################################################################
# Defining Data Loader
# --------------------
#
# A graph classification dataset usually contains two types of elements: a
# set of graphs, and their graph-level labels. Similar to an image
# classification task, when the dataset is large enough, we need to train
# with mini-batches. When you train a model for image classification or
# language modeling, you will use a ``DataLoader`` to iterate over the
# dataset. In DGL, you can use the ``GraphDataLoader``.
#
# You can also use various dataset samplers provided in
# `torch.utils.data.sampler <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`__.
# For example, this tutorial creates a training ``GraphDataLoader`` and
# test ``GraphDataLoader``, using ``SubsetRandomSampler`` to tell PyTorch
# to sample from only a subset of the dataset.
#

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


######################################################################
# You can try to iterate over the created ``GraphDataLoader`` and see what it
# gives:
#


it = iter(train_dataloader)
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
print(graphs)


######################################################################
# Define Model
# ------------
#
# This tutorial will build a two-layer `Graph Convolutional Network
# (GCN) <http://tkipf.github.io/graph-convolutional-networks/>`__. Each of
# its layer computes new node representations by aggregating neighbor
# information. If you have gone through the
# :doc:`introduction <1_introduction>`, you will notice two
# differences:
#
# -  Since the task is to predict a single category for the *entire graph*
#    instead of for every node, you will need to aggregate the
#    representations of all the nodes and potentially the edges to form a
#    graph-level representation. Such process is more commonly referred as
#    a *readout*. A simple choice is to average the node features of a
#    graph with ``dgl.mean_nodes()``.
#
# -  The input graph to the model will be a batched graph yielded by the
#    ``GraphDataLoader``. The readout functions provided by DGL can handle
#    batched graphs so that they will return one representation for each
#    minibatch element.
#

from dgl.nn import GraphConv


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


######################################################################
# Training Loop
# -------------
#
# The training loop iterates over the training set with the
# ``GraphDataLoader`` object and computes the gradients, just like
# image classification or language modeling.
#

# Create the model with given dimensions
model = GCN(3, 16, dataset.num_classes)
print("OK")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print("OK2")
for epoch in range(20):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata["node_attr"].float())
        #print("batched_graph: ",batched_graph)
        #print("batched_graph.ndata[attr] ", batched_graph.ndata["node_attr"].float())
        #print("pred: ",pred)
        #print("label: ",labels)
        loss = F.cross_entropy(pred, labels.squeeze(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct = 0
num_tests = 0
elapsed_time = 0
for batched_graph, labels in test_dataloader:
    start = time.time()
    pred = model(batched_graph, batched_graph.ndata["node_attr"].float())
    end = time.time()
    elapsed_time = elapsed_time + (end - start)
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)*5
print("num_correct: ",num_correct)
print("num_tests: ",num_tests)
print("Test accuracy:", num_correct / num_tests)
print("Infer time:", elapsed_time)


######################################################################
# What’s next
# -----------
#
# -  See `GIN
#    example <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin>`__
#    for an end-to-end graph classification model.
#


# Thumbnail credits: DGL
# sphinx_gallery_thumbnail_path = '_static/blitz_5_graph_classification.png'

