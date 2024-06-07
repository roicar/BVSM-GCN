# FileFolders (part of result of expr)
B ------------------------------ TVM*-B
G ------------------------------ TVM*-G(modifying...)
M ------------------------------ TVM*-M
Opt ---------------------------- effect of Opt of TVM complier level
Sparse ------------------------- TVM*-Sparse
dense -------------------------- TVM*-dense
dense_rerange ------------------ TVM*-dense + compute rerange
Deprecated --------------------- deprecated code


# TVM Tutorial
build_gcn.py

# TVM*-dense
Relay_infer_sparse.py ---------------------------- relay.nn.sparse_dense(original)
Relay_infer_dense.py ----------------------------- relay.nn.dense
Relay_infer_dense_nT.py -------------------------- relay.nn.batch_matmul
Relay_infer_dense_nT_DAD.py ---------------------- preprocess inculdes DAD


# TVM+BVSM
Relay_infer_BVSM_B.py ---------------------------- TVM*-B
Relay_infer_BVSM_M.py ---------------------------- TVM*-M
Relay_infer_BVSM_G.py ---------------------------- TVM*-G


# TVM+BVSM+Auto-scheduler
Ansor.py ----------------------------------------- TVM using Ansor code
Relay_infer_dense_Ansor.py ----------------------- TVM+Ansor+GCN
Relay_infer_BVSM_B*.py --------------------------- TVM+BVSM-B(modifying...)
Relay_infer_BVSM_M*.py --------------------------- TVM+BVSM-M(modifying...)
Relay_infer_BVSM_G*.py --------------------------- TVM+BVSM-G(modifying...)
