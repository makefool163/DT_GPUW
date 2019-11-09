from __future__ import absolute_import
from __future__ import print_function

import pyopencl as cl
import pyopencl.array as cl_array

import os
import codecs
import numpy,time
import numba, datetime

@numba.njit(fastmath=True,parallel=True,nogil=True)
def dtw_1D_jit2(s1,s2):
    l1 = len(s1)
    l2 = len(s2)
    cum_sum = numpy.empty((l1 + 1, l2 + 1))
    cum_sum[0,  0] = 0.0
    cum_sum[1:, 0] = numpy.inf
    cum_sum[0, 1:] = numpy.inf

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = (s1[i]-s2[j])*(s1[i]-s2[j])

    for i in range(l1):
        for j in range(l2):
            #cum_sum[i + 1, j + 1] = (s1[i]-s2[j])*(s1[i]-s2[j])
            if numpy.isfinite(cum_sum[i + 1, j + 1]):
                cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j])
    ret = numpy.sqrt(cum_sum[l1, l2])
    return (ret)

def cpu_dtw (SrcS, TrgS, funC):
    ret = numpy.empty ((SrcS.shape[0],TrgS.shape[0]))
    print ('ret shape',ret.shape)
    for i in range(SrcS.shape[0]):
        for j in range(TrgS.shape[0]):
            a = SrcS[i]
            b = TrgS[j]
            ret[i,j] = funC(a,b)            
    return ret

def opencl_dtw (SrcS, TrgS, show=True):
    #func = cuda_dtw_prepare ()
    t0 = time.time()
    ctx, queue, prg, dev_Param = OpenCL_Init ()
    ret = opencl_dtw_run (SrcS, TrgS, ctx, queue, prg, dev_Param)
    if show:
        print ("opencl run time:  ",time.time()-t0)
    return (ret)

def OpenCL_Init():
    fp = codecs.open("./opencldtw.cl","r","utf-8")
    opencl_Source_Str = fp.read()
    fp.close()

    ctx = cl.create_some_context()
    prg = cl.Program(ctx, opencl_Source_Str).build()

    dev_Param = {}
    dev_Param["MAX_MEM_ALLOC_SIZE"] = 1024*1024*64
    dev_Param["LOCAL_MEM_SIZE"] = 1024*8
    dev_Param["MAX_WORK_ITEM_SIZES"] = [512,512]
    dev_Param["MAX_WORK_GROUP_SIZE"] = 512
    dev_Param["MAX_WORK_GROUP_SIZE"] = 512


    dev_Param["MAX_MEM_ALLOC_SIZE"] = \
        ctx.devices[0].get_info(getattr(cl.device_info, "MAX_MEM_ALLOC_SIZE"))
    dev_Param["MAX_MEM_ALLOC_SIZE"] = \
        dev_Param["MAX_MEM_ALLOC_SIZE"] - 64*1024*1024
    dev_Param["LOCAL_MEM_SIZE"] = \
        ctx.devices[0].get_info(getattr(cl.device_info, "LOCAL_MEM_SIZE"))
    dev_Param["MAX_WORK_ITEM_SIZES"] = \
        ctx.devices[0].get_info(getattr(cl.device_info, "MAX_WORK_ITEM_SIZES"))
    dev_Param["MAX_WORK_GROUP_SIZE"] = \
        ctx.devices[0].get_info(getattr(cl.device_info, "MAX_WORK_GROUP_SIZE"))
    #print (dev_Param)

    queue = cl.CommandQueue(ctx)
    return ctx, queue, prg, dev_Param

def opencl_dtw_run (SrcS, TrgS, ctx, queue, prg, dev_Param):
    #MAX_MEM_ALLOC_SIZE
    cot1 = int(dev_Param["LOCAL_MEM_SIZE"] / (TrgS.shape[1] *4 *3))
    cot2 = int(dev_Param["MAX_WORK_ITEM_SIZES"][0] / TrgS.shape[1])
    TRG_COT = min(cot1, cot2)
    Grp_Cot = int(dev_Param["MAX_MEM_ALLOC_SIZE"] / (TrgS.shape[1] *4 *TRG_COT))

    T0 = TrgS.shape[0]
    T1 = TrgS.shape[1]

    TrgS_Alignment =  TRG_COT -T0 % TRG_COT
    if TrgS_Alignment != TRG_COT:
        TrgS = numpy.concatenate ((TrgS, numpy.ones((TrgS_Alignment,T1),dtype=numpy.float32)))
    T0 = TrgS.shape[0]
    #print ("TrgS_Alignment,TRG_COT",TrgS_Alignment,TRG_COT)

    Splits = list(range(0, T0, Grp_Cot *TRG_COT))
    Splits.append (T0)
    allret = numpy.empty ((SrcS.shape[0],TrgS.shape[0]), dtype=numpy.float32)

    for j in range(len(Splits)-1):
        TrgS_sub = TrgS[Splits[j]:Splits[j+1],:]
        Ts0 = TrgS_sub.shape[0]
        Ts1 = TrgS_sub.shape[1]
        local_size  = TRG_COT *Ts1
        global_size = Ts0 *Ts1
        
        t = numpy.reshape(TrgS_sub,(Ts0 *Ts1))
        t_dev = cl_array.to_device(queue, t)
        #print ("local_size, global_size ",local_size,global_size,t.nbytes/1024/1024)
 
        SRC_LEN = SrcS.shape[1]
        TRG_LEN = TrgS.shape[1]

        for i in range(SrcS.shape[0]):
            s = SrcS[i,:]
            s_dev = cl_array.to_device(queue, s)
            r_dev = cl_array.empty (queue, (Ts0,), dtype=numpy.float32)
            shared_mem_size = Ts1 *TRG_COT *4

            prg.opencl_dtw(queue, (global_size,), (local_size,), \
                numpy.uint32(SRC_LEN),numpy.uint32(TRG_LEN),numpy.uint32(TRG_COT),
                s_dev.data, t_dev.data, r_dev.data,\
                cl.LocalMemory(shared_mem_size),
                cl.LocalMemory(shared_mem_size),
                cl.LocalMemory(shared_mem_size)
                )
            r = r_dev.get()
            allret[i,Splits[j]:Splits[j+1]] = r
            #print(la.norm((dest_dev - (a_dev+b_dev)).get()))

    if TrgS_Alignment != TRG_COT:
        allret = allret[:,0:-TrgS_Alignment]
    return (allret)

if __name__ == '__main__':
    zz0 = numpy.random.random ((1,46))
    zz0 = zz0.astype(numpy.float32)
    zz1 = numpy.random.random ((1024*1024*1+1234,46))
    zz1 = zz1.astype(numpy.float32)
    print ("zz1.size ",zz1.nbytes /1024/1024, zz1.shape)
    
    #os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    #os.environ['PYOPENCL_CTX'] = '0'

    ret = opencl_dtw (zz0, zz1)
    print ("ret\n",ret.shape,ret[-1,-10:],ret[0,:10])

    t0 = time.time()
    ret_cpu =cpu_dtw (zz0,zz1, dtw_1D_jit2)
    print ("cpu\n",time.time()-t0)
    print ("ret \n",ret_cpu.shape,ret_cpu[-1,-10:],ret_cpu[0,:10])
"""
    t0 = time.time()
    ret = cuda_dtw (zz0, zz1)
    print ("cuda\n",time.time()-t0)
    print ("ret\n",ret.shape,ret[-1,-10:],ret[0,:10])
"""
