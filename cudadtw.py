from __future__ import absolute_import
from __future__ import print_function

import pycuda.driver as cuda_drv
from pycuda.compiler import SourceModule
import pycuda.autoinit

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

def cuda_dtw_prepare ():
    fp = codecs.open("./cudadtw.cu","r","utf-8")
    cuda_Source_Str = fp.read()
    fp.close()
    mod = SourceModule(cuda_Source_Str)
    func_calc_dtw = mod.get_function("calc_dtw")
    return func_calc_dtw

def gen_Split_array (X, A, b):
    ret = list(range(0,X,A*b))
    left_X = X % (A*b)
    if left_X == 0:
        ret.append (X)
    else:
        if int(left_X/b) > 0:
            ret.append (ret[-1]+int(left_X/b)*b)
        left_X = X % b
        if left_X != 0:
            ret.append (X)
    return (ret)

def get_CUDA_Param ():
    GPU_Param = {}
    GPU_Param["Max_Grid_X"] = 65535
    GPU_Param["Max_Grid_Y"] = 65535
    GPU_Param["Max_Threads_pre_Block"] = 512
    GPU_Param["Max_share_memeory_per_block"] = 8*1024
    GPU_Param["COMPUTE_CAPABILITY_MAJOR"] = 1

    dev = cuda_drv.Device(0)
    GPU_Param["total_memory"] = dev.total_memory()
    dev_attrib = {}
    for att, value in dev.get_attributes().items():
        dev_attrib[str(att)] = value
    if "COMPUTE_CAPABILITY_MAJOR" in dev_attrib.keys():
        GPU_Param["COMPUTE_CAPABILITY_MAJOR"] = dev_attrib["COMPUTE_CAPABILITY_MAJOR"]
    if "MAX_GRID_DIM_X" in dev_attrib.keys():
        GPU_Param["Max_Grid_X"] = dev_attrib["MAX_GRID_DIM_X"]
    #GPU_Param["Max_Grid_X"] = 4096

    if "MAX_GRID_DIM_Y" in dev_attrib.keys():
        GPU_Param["Max_Grid_Y"] = dev_attrib["MAX_GRID_DIM_Y"]
    if "MAX_THREADS_PER_BLOCK" in dev_attrib.keys():
        GPU_Param["Max_Threads_pre_Block"] = dev_attrib["MAX_THREADS_PER_BLOCK"]
    if "MAX_SHARED_MEMORY_PER_BLOCK" in dev_attrib.keys():
        GPU_Param["Max_share_memeory_per_block"] = dev_attrib["MAX_SHARED_MEMORY_PER_BLOCK"]
    print (GPU_Param)
    return GPU_Param

#@numba.jit
def cuda_dtw_run (SrcS, TrgS, func_calc_dtw, GPU_Param):
    Block_Trg_Count1 = int(GPU_Param["Max_Threads_pre_Block"] / TrgS.shape[1])
    Block_Trg_Count2 = int(GPU_Param["Max_share_memeory_per_block"] / (3*TrgS.shape[1]*4))
    Block_Trg_Count = min (Block_Trg_Count1,Block_Trg_Count2)

    T0 = TrgS.shape[0]
    T1 = TrgS.shape[1]

    TrgS_Alignment =  Block_Trg_Count -T0 % Block_Trg_Count
    if TrgS_Alignment != Block_Trg_Count:
        TrgS = numpy.concatenate ((TrgS, numpy.ones((TrgS_Alignment,T1),dtype=numpy.float32)))
    #print ("TrgS_Alignment",TrgS_Alignment,TrgS.shape[0])
    T0 = TrgS.shape[0]

    blockDim_x = T1 *Block_Trg_Count
    if GPU_Param["total_memory"] > 4*1024*1024*1024:
        MemoryUsing = 4*1024*1024*1024 - 150*1024*1024
    else:
        if GPU_Param["COMPUTE_CAPABILITY_MAJOR"] <= 2:
            if GPU_Param["total_memory"] > 128*1024*1024:
                MemoryUsing = 160*1024*1024
            else:
                MemoryUsing = 64*1024*1024
        else:
            MemoryUsing = int(GPU_Param["total_memory"] - 150*1024*1024)
         
    # ---
    #MemoryUsing = 512*1024*1024
    # ---
    gridDim_X = int (MemoryUsing /(Block_Trg_Count*T1*4))
    if gridDim_X > GPU_Param["Max_Grid_X"]:
        gridDim_X = GPU_Param["Max_Grid_X"]
    gridDim_Y = int (MemoryUsing /(gridDim_X *Block_Trg_Count* T1*4))
    if gridDim_Y > GPU_Param["Max_Grid_Y"]:
        gridDim_Y = GPU_Param["Max_Grid_Y"]
    if gridDim_Y == 0:
        gridDim_Y = 1
    
    Splits = list(range(0,T0,gridDim_X*gridDim_Y*Block_Trg_Count))
    if T0 %(gridDim_X*gridDim_Y*Block_Trg_Count) == 0:
        Splits.append (T0)
    else:
        t_Splits = list(range(Splits[-1], T0, gridDim_X*Block_Trg_Count))
        if len(t_Splits) > 1:
            Splits.append(Splits[-1] + (len(t_Splits)-1)*gridDim_X*Block_Trg_Count)
        if T0 %(gridDim_X*Block_Trg_Count) == 0:
            Splits.append (T0)
        else:
            t_Splits = list(range(Splits[-1], T0, Block_Trg_Count))
            if len(t_Splits) > 1:
                Splits.append(Splits[-1] + len(t_Splits)*Block_Trg_Count)

    print ("grids,block_DimX",gridDim_X,gridDim_Y,Block_Trg_Count,blockDim_x)
    print ("Splits",Splits)
    allret = numpy.ones ((SrcS.shape[0],T0))

    t0 = time.time()
    for j in range(len(Splits)-1):
        TrgS_sub = TrgS[Splits[j]:Splits[j+1],:]
        Gy = int(TrgS_sub.shape[0]/(Block_Trg_Count *gridDim_X))
        if Gy == 0:
            Gy = 1
            Gx = int(TrgS_sub.shape[0] /Block_Trg_Count)
        else:
            Gx = int(TrgS_sub.shape[0] /(Block_Trg_Count*Gy))
        t = numpy.reshape(TrgS_sub,(TrgS_sub.shape[0]*TrgS_sub.shape[1]))

        print ("grid_run T0,Gx,Gy,Tbits ",TrgS_sub.shape[0],Gx,Gy,t.nbytes/1024/1024)
        #gpu_t = cuda_drv.mem_alloc(t.nbytes)
        #cuda_drv.memcpy_htod(gpu_t, t)

        share_memory_size = int(3*Block_Trg_Count*TrgS_sub.shape[1]*4)
        #print ("share_memory_size ", share_memory_size)

        r = numpy.ones(Block_Trg_Count*Gx*Gy,dtype=numpy.float32)
        gpu_r = cuda_drv.mem_alloc(r.nbytes)
        cuda_drv.memcpy_htod(gpu_r, r)

        for i in range(SrcS.shape[0]):
            s = SrcS[i,:]
            func_calc_dtw(
                numpy.uint32(SrcS.shape[1]) ,\
                numpy.uint32(TrgS_sub.shape[1]) ,\
                numpy.uint32(Block_Trg_Count) ,\
                cuda_drv.In(s), cuda_drv.In(t), \
                gpu_r,\
                shared = share_memory_size,\
                block=(blockDim_x,1,1), \
                grid=(Gx,Gy,1))
            cuda_drv.Context.synchronize()
            cuda_drv.memcpy_dtoh (r, gpu_r)
            #cuda_drv.memcpy_dtoh_async (r, gpu_r)

            # HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
            # TdrLevel 0
            # TdrDelay 12000000

            allret[i,Splits[j]:Splits[j+1]] = r
        """
        if i%50 == 0:
            print (i,"\t",datetime.datetime.now(),time.time()-t0)
            t0 = time.time()
        """
    if TrgS_Alignment != 0:
        return allret[:,:-TrgS_Alignment]
    else:
        return allret

def cuda_dtw (SrcS, TrgS):
    t0 = time.time()
    func = cuda_dtw_prepare ()
    GPU_Param = get_CUDA_Param ()
    ret = cuda_dtw_run (SrcS, TrgS, func,GPU_Param)
    print ("cuda runtime:  ",time.time()-t0)
    return (ret)

if __name__ == '__main__':
    zz0 = numpy.random.random ((1,46))
    zz0 = zz0.astype(numpy.float32)
    zz1 = numpy.random.random ((1024*1024*1+1234,46))
    zz1 = zz1.astype(numpy.float32)
    print ("zz1.size ",zz1.nbytes /1024/1024, zz1.shape)
    
    #os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    #os.environ['PYOPENCL_CTX'] = '0'

    ret = cuda_dtw (zz0, zz1)
    print ("ret\n",ret.shape,ret[-1,-10:],ret[0,:10])

    t0 = time.time()
    ret_cpu =cpu_dtw (zz0,zz1, dtw_1D_jit2)
    print ("cpu\n",time.time()-t0)
    print ("ret \n",ret_cpu.shape,ret_cpu[-1,-10:],ret_cpu[0,:10])

"""
    t0 = time.time()
    ret = opencl_dtw (zz0, zz1)
    print ("opencl\n",time.time()-t0)
    print ("ret\n",ret.shape,ret[-1,-10:],ret[0,:10])
"""