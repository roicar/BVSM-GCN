# -*- coding:utf8 -*
import logging
import numpy as np
import tvm
import sys
import math
import timeit
from tvm import autotvm
from tvm import te
import time

def numpyBaseline(M, K, N):
    np_repeat = 100
    np_runing_time = timeit.timeit(setup='import numpy\n'
                                         'M = ' + str(M) + '\n'
                                                           'K = ' + str(K) + '\n'
                                                                             'N = ' + str(N) + '\n'
                                                                                               'dtype = "float32"\n'
                                                                                               'a = numpy.random.rand(M, K).astype(dtype)\n'
                                                                                               'b = numpy.random.rand(K, N).astype(dtype)\n',
                                   stmt='answer = numpy.dot(a, b)',
                                   number=np_repeat)
    print("Numpy running time: %f" % (np_runing_time / np_repeat))


def mytuner(n_trial, M, K, N, gemm_impl_schedule, early_stopping):

    target = tvm.target.Target(target="llvm -mcpu=core-avx2")
    dtype = 'float32'
    ctx = tvm.device(target.kind.name, 0)
    src = str(M) + "x" + str(K) + "x" + str(N)
    recordFileName = gemm_impl_schedule.__name__+'_XGBTuner_matmul_' + src + 'tuner00.log'
    print(src)
    numpyBaseline(M,K,N)
    tsk = autotvm.task.create(gemm_impl_schedule.__name__, args=(M, K, N, dtype), target=target)
    n_trial = min(n_trial, len(tsk.config_space))
    #print(tsk.config_space)
    measure_option = autotvm.measure_option(builder='local', runner=autotvm.LocalRunner(number=10))

    XGBtuner = autotvm.tuner.XGBTuner(tsk)
    XGBtuner.tune(n_trial=n_trial,
                   early_stopping=early_stopping,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.progress_bar(n_trial), autotvm.callback.log_to_file(recordFileName)])

    autotvm.record.pick_best(recordFileName, recordFileName + ".best")
    with autotvm.apply_history_best(recordFileName + ".best"):
        with tvm.target.Target("llvm -mcpu=core-avx2"):
             s, arg_bufs = mm(M, K, N, "float32")
             print("arg_bufs:",arg_bufs)
             func = tvm.build(s, arg_bufs)
             a_np = np.random.rand(M, K).astype(dtype)
             b_np = np.random.rand(K, N).astype(dtype)
             a_tvm = tvm.nd.array(a_np, ctx)
             b_tvm = tvm.nd.array(b_np, ctx)
             c_np = a_np.dot(b_np)
             c_tvm = tvm.nd.empty(c_np.shape)
             
             

             func(a_tvm,b_tvm,c_tvm)
             evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 100, min_repeat_ms=500)
             rel = evaluator(a_tvm, b_tvm, c_tvm).results
             print(
            "Execution time of this operator: max:%.3f ns   median:%.3f ns   min:%.3f ns"
            % (np.max(rel) * 1000000000, np.median(rel) * 1000000000, np.min(rel) * 1000000000)
        )
@autotvm.template("mm")
def mm(M, K, N, dtype):
    data = te.placeholder((M, K), name='data', dtype=dtype)
    weight = te.placeholder((N, K), name='weight', dtype=dtype)
    k = te.reduce_axis((0, K), name='k')
    out = te.compute((M, N), lambda i, j: te.sum(data[i, k] * weight[k, j], axis=k), name='out')
    s = te.create_schedule(out.op)
    
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=8)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=1)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=8)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=1)
    s[out].reorder(out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_i_j_o_o_i_fused = s[out].fuse(out_i_o_o_i, out_j_o_o_i)
    
    s[out].pragma(out_i_o_o_i_j_o_o_i_fused, "auto_unroll_max_step", 64)
    s[out].pragma(out_i_o_o_i_j_o_o_i_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    #BVSM_S
    z = s[out].op.reduce_axis[0]
    x, y = s[out].op.axis
    cfg = autotvm.get_config()
    cfg.define_split("tile_x", x, num_outputs=3)
    cfg.define_split("tile_y", y, num_outputs=3)
    cfg.define_split("tile_z", z, num_outputs=2)
    xt, xo, xi = cfg["tile_x"].apply(s, out, x)
    yt, yo, yi = cfg["tile_y"].apply(s, out, y)
    zo, zi = cfg["tile_z"].apply(s, out, z)
    s[out].reorder(xt, yt, zo, xo, yo, zi, xi, yi)
    out_fused = s[out].fuse(xt,yt)
    #s[out].parallel(out_fused)
    s[out].pragma(out_fused, "auto_unroll_max_step", 2048)
    s[out].pragma(out_fused, "unroll_explicit", True)
    s[out].vectorize(yi)
    
    #18_18_18
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=1)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=3)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=6)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=6)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=3)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=6)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_j_o_o_o_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o)
    s[out].parallel(out_i_o_o_o_j_o_o_o_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused, "auto_unroll_max_step", 64)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    
    
    #18_18_18_pro1
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=1)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=3)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=6)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=18)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=1)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=6)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_j_o_o_o_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o)
    s[out].parallel(out_i_o_o_o_j_o_o_o_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused, "auto_unroll_max_step", 64)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=1)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=3)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=6)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=9)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=2)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=6)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_j_o_o_o_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o)
    s[out].parallel(out_i_o_o_o_j_o_o_o_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused, "auto_unroll_max_step", 64)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    #18_18_18_pro2
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=1)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=3)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=6)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=6)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=3)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=6)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_j_o_o_o_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o)
    s[out].parallel(out_i_o_o_o_j_o_o_o_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused, "auto_unroll_max_step", 1)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    #18_18_18_pro3
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=3)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=6)
    
    out_j_o_i, out_j_i = s[out].split(out_j, factor=6)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=3)
    
    
    s[out].reorder(out_i_o_i, out_j_o_i, out_k, out_i_i, out_j_i)
    
    s[out].pragma(out_i_o_i, "auto_unroll_max_step", 64)
    s[out].pragma(out_i_o_i, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    #18_18_18_pro4
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=3)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=6)
    
    out_j_o_i, out_j_i = s[out].split(out_j, factor=6)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=3)
    
    out_k_o, out_k_i = s[out].split(out_k, factor=6)
    s[out].reorder(out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    
    s[out].pragma(out_i_o_i, "auto_unroll_max_step", 128)
    s[out].pragma(out_i_o_i, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    
    #20_20_20
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=4)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=5)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=1)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=20)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=1)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=20)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i)
    s[out].parallel(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused, "auto_unroll_max_step", 512)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    #20_20_20_pro1
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=4)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=5)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=1)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=20)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=1)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=20)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i)
    s[out].parallel(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused, "auto_unroll_max_step", 1)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    #20_20_20_pro2
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=4)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=5)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=1)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=1)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=20)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=20)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i)
    s[out].parallel(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused, "auto_unroll_max_step", 512)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    #20_20_20_pro3
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=4)  
    s[out].reorder(out_i_o_i, out_k, out_i_i, out_j)
    s[out].pragma(out_i_o_i, "auto_unroll_max_step", 512)
    s[out].pragma(out_i_o_i, "unroll_explicit", True)
    s[out].vectorize(out_j)
    '''
    
    #21_21_21
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=3)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=1)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=7)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=7)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=3)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=1)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o = s[out].fuse(out_i_o_o_o)
    s[out].parallel(out_i_o_o_o)
    s[out].pragma(out_i_o_o_o, "auto_unroll_max_step", 512)
    s[out].pragma(out_i_o_o_o, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=4)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=5)
    #out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=1)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=20)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=1)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=20)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i)
    s[out].parallel(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused, "auto_unroll_max_step", 512)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused_j_o_o_i_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    
    #32_32_32 SSRSRS
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=8)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=1)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=4)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=8)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=4)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=32)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_fused = s[out].fuse(out_i_o_o_o)
    s[out].parallel(out_i_o_o_o_fused)
    s[out].pragma(out_i_o_o_o_fused, "auto_unroll_max_step", 0)
    s[out].pragma(out_i_o_o_o_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    #32_32_32 RSRS
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=8)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=8)
    out_k_o, out_k_i = s[out].split(out_k, factor=32)
    s[out].reorder(out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_fused = s[out].fuse(out_k_o)
    s[out].pragma(out_i_o_o_o_fused, "auto_unroll_max_step", 0)
    s[out].pragma(out_i_o_o_o_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    #64_64_64 SSRSRS
    '''
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=4)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=1)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=16)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=16)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=1)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
    out_k_o, out_k_i = s[out].split(out_k, factor=4)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i)
    s[out].parallel(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused, "auto_unroll_max_step", 16)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    
    '''
    #64_64_64 SRSRS
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    #out_i_o_i, out_i_i = s[out].split(out_i, factor=4)
    #out_j_o_i, out_j_i = s[out].split(out_j, factor=16)
    #out_k_o, out_k_i = s[out].split(out_k, factor=4)
    s[out].reorder(out_k, out_i, out_j)
    out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused = s[out].fuse(out_k)
    #s[out].parallel(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused, "auto_unroll_max_step", 64)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused, "unroll_explicit", True)
    s[out].vectorize(out_j)
    '''
    '''
    #128 128 128 SSRSRS
    out_i, out_j, out_k = tuple(out.op.axis) + tuple(out.op.reduce_axis)
    out_i_o_i, out_i_i = s[out].split(out_i, factor=2)
    out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=1)
    out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=8)
    out_j_o_i, out_j_i = s[out].split(out_j, factor=32)
    out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=1)
    out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=4)
    out_k_o, out_k_i = s[out].split(out_k, factor=16)
    s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_k_o, out_i_o_i, out_j_o_i, out_k_i, out_i_i, out_j_i)
    out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i)
    s[out].parallel(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused, "auto_unroll_max_step", 0)
    s[out].pragma(out_i_o_o_o_j_o_o_o_fused_i_o_o_i_fused, "unroll_explicit", True)
    s[out].vectorize(out_j_i)
    '''
    #print("Lowered TIR:")
    #print(tvm.lower(s, [data, weight, out], simple_mode=True))
    return s, [data, weight, out]
    
    
    
    
    
    
if __name__ == '__main__':
    n_trial = 1000

    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])
    mytuner(n_trial, M, K, N, mm, 400)
'''

'''
