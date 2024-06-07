runtimes:
n_trials = 10

unit:
ms

run method(example as aspirin):
cd ~/model_Batched_GEMM
python Pytorch_batchMM.py aspirin 10 | tee Pytorch_batchMM_aspirin.txt
