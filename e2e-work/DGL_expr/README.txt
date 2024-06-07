# FileFolders related expr results
SparseDirect ---------------------- DGL origin Sparse method
Batch ----------------------------- DGL origin Sparse(batch) method
Sparse ---------------------------- modified DGL Sparse method 
Dense ----------------------------- modified DGL Dense method
Parameters ------------------------ model parameters used before(You may need to try the latest parameters if you can)
Deprecated ------------------------ some python scripts deprecated

# DGL+Sparse
DGL.py ----------------------- GraphConv of DGL type
SpMM-DGL-1.py ---------------- SparseGCNConv + preNormalize(DAD) + nn.linear
SpMM-DGL-nolinear.py --------- SparseGCNConv + preNormalize(DAD)
SpMM-DGL-nolinear-vec.py ----- SparseGCNConv (+ timing)
SpMM-DGL-nolinear-withouttime.py ------SparseGCNConv


# DGL+Dense
D-SpMM-DGL.py ---------------- torch.matmul + nn.linear + preNormalize(DAD)
DenseMM-DGL.py --------------- np.dot + nn.linear + preNormalize(DAD)
DenseMM-DGL-nolinear.py ----------------------- np.dot + preNormalize(DAD)
DenseMM-DGL-nolinear-choose.py ---------------- np.dot + preNormalize(DAD) + associative law
DenseMM-DGL-nolinear-DH.py -------------------- np.dot
