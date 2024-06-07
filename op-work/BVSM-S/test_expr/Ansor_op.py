import logging
#testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore
import time
import timeit
import math
import sys
import numpy as np
import tvm
from tvm import testing
from tvm import te, auto_scheduler




@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul(M, K, N, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    out = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="out",
        #attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    

    return [A, B, out]



def mytuner(n_trial, M, K, N, early_stopping):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    #target = tvm.target.Target("llvm -keys=cpu -link-params=0 -mcpu=core-avx2“）
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(M, K, N, "float32"), target=target)
    
    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)


    log_file = "matmul.json"
    tune_option = auto_scheduler.TuningOptions(
        early_stopping = 400,
        num_measure_trials=n_trial,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )



    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best("matmul.json")


    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))



    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    
    out_np = a_np.dot(b_np)

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)

    out_tvm = tvm.nd.empty(out_np.shape, device=dev)


    func(a_tvm, b_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)


    evaluator = func.time_evaluator(func.entry_name, dev, repeat = 100, min_repeat_ms=500)
    rel = evaluator(a_tvm, b_tvm, out_tvm).results
    print(
        "Execution time of this operator: max:%.3f us   median:%.3f us   min:%.3f us"
        % (np.max(rel) * 1000000, np.median(rel) * 1000000, np.min(rel) * 1000000)
    )



    print("Equivalent python schedule:")
    print(task.print_best(log_file))




    
if __name__ == '__main__':
    n_trial = 500
    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])
    mytuner(n_trial, M, K, N, 400)
    
