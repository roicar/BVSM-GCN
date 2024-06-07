import pandas as pd
import os
import networkx as nx
from sklearn import preprocessing
import numpy as np
import sys

'''
给定DS数据集，将原始数据txt转化成
几个不同需求的csv文件，方便后续矩阵乘法
writted by Song
'''

def Alter(DS):
    #DS = DS  # The filedirname of the processed dataset (ENZYMES as an example)
    # Read DS_A, DS_graph_indicator, DS_node_attributes
    A = pd.read_csv('./DataSets' + '/' + DS + '/' + DS + '_A.txt', delimiter=', ', header=None)
    graph_indicator = pd.read_csv('./DataSets' + '/' + DS + '/' + DS + '_graph_indicator.txt', header=None).values
    node_attributes = pd.read_csv('./DataSets' + '/' + DS + '/' + DS + '_node_attributes.txt', header=None)
    n = node_attributes.shape[1]  # 3rd Dim of SpMM
    iterEnd = len(graph_indicator)

    # Final output files to be found under './DS/result/':
    #     sparseA_m_k_n_graphIdx.csv
    #     denseB_m_k_n_graphIdx.csv
    #     denseC_m_k_n_graphIdx.csv
    outputfiledir = './DataSets' + '/' + DS + '/' + 'result/'
    os.mkdir('./DataSets' + '/' + DS + '/' + 'result')  # If toggled, neccessary to create manully

    prev = 1
    gBGN = 0
    # Generate and save the information of subgraphs
    for i in range(iterEnd):
        cur = graph_indicator[i]
        if cur != prev or i == iterEnd - 1:
            # Obtain the sparse matrix A

            edge_list = A[(A[0] > gBGN) & (A[0] <= i) & (A[1] > gBGN) & (A[1] <= i)].reset_index(drop=True)
            g = nx.Graph()
            g.add_nodes_from(np.arange(gBGN + 1, i + 1))
            g.add_edges_from(tuple(x) for x in edge_list.values)
            m = g.number_of_nodes()  # 1st Dim of SpMM
            k = m  # 2nd Dim of SpMM
            # A^{\tilde} = A + I
            a = nx.adjacency_matrix(g).todense() + np.diag([1] * m)

            # Calculate D^{\tilde}_{in}, D^{\tilde}_{out}
            d_in_n_0_5 = np.mat(np.diag(np.power(a.sum(1).T, -0.5).tolist()[0]))
            d_out_n_0_5 = np.mat(np.diag(np.power(a.sum(0), -0.5).tolist()[0]))

            # Calculate A^{sp} = D^{\tilde}_{in} \times A^{\tilde} \times D^{\tilde}_{out}
            SpM = np.dot(d_in_n_0_5, np.dot(a, d_out_n_0_5))

            # Obtain the node features matrix H^{(0)}

            DeM = node_attributes.iloc[gBGN:i, :]

            # normalize the feature matrix
            stdnorm = preprocessing.StandardScaler()
            DeM = stdnorm.fit_transform(DeM)

            # Save sparseA, denseB and denseC (st. C = A * B)
            np.savetxt(outputfiledir + 'sparseA_' + str(m) + '_' + str(k) + '_' + str(n) + '_' + str(
                graph_indicator[i - 1][0]) + '.csv',
                       SpM, delimiter=',')
            np.savetxt(outputfiledir + 'denseB_' + str(m) + '_' + str(k) + '_' + str(n) + '_' + str(
                graph_indicator[i - 1][0]) + '.csv',
                       DeM, delimiter=',')
            ResM = np.dot(SpM, DeM)
            np.savetxt(outputfiledir + 'denseC_' + str(m) + '_' + str(k) + '_' + str(n) + '_' + str(
                graph_indicator[i - 1][0]) + '.csv',
                       ResM, delimiter=',')
            prev = cur
            gBGN = i

if __name__ == '__main__':
    DS = str(sys.argv[1])
    Alter(DS)