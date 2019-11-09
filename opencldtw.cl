__kernel void opencl_dtw (
unsigned SRC_LEN,
unsigned TRG_LEN,
unsigned TRG_COT, 
__global const float *S,
__global const float *TT,
__global float       *RR,
__local float path_h1[],
__local float path_h2[],
__local float dist[]
)
{
    /*
    __local float* path_h1 = (float*)array;
    __local float* path_h2 = (float*)&path_h1[blockDim.x];
    __local float* dist    = (float*)&path_h2[blockDim.x];
    */

    //int blockId = (blockIdx.y*gridDim.x + blockIdx.x);
    //int G_idx   = blockId *blockDim.x +threadIdx.x;

    //float *T =     TT +get_group_id(0) *get_local_size(0);
    //float *R = Result +get_group_id(0) *TRG_COT;

    __local float *ex;
    
    int t_gid = get_group_id(0) *get_local_size(0);
    //int r_gid = get_group_id(0) *TRG_COT;
    
    int i,j;    
    int blk_id = get_local_id(0);
    int sub_x  = blk_id % TRG_LEN;
    //int x_cot = get_local_id(0) / TRG_LEN;
    
    // first line speical, do first
    // 1. paralle, first line's every element's dist
    dist[blk_id] = (S[0] -TT[get_global_id(0)])
                  *(S[0] -TT[get_global_id(0)]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. serie, first line's every element's serie's dist
    if (sub_x == 0){
        path_h1[blk_id] = dist[blk_id];
        for (i=1; i <TRG_LEN; i++) {
            path_h1[i +blk_id] = path_h1[i-1 +blk_id]
                               +    dist[i   +blk_id];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /*
    if (get_local_id(0) == 0)
        printf ("group_0 %d ,local_size %d\n", get_group_id(0) ,get_local_size(0));
    */

    for (i=1; i <SRC_LEN; i++){ // do circle
        // 1. paralle, calc itself's DISTANCE, for speed follow progress
        //    use memeory to rise speed
        dist[blk_id] = (S[i] -TT[get_global_id(0)])
                      *(S[i] -TT[get_global_id(0)]);
        barrier(CLK_LOCAL_MEM_FENCE);
        // 2. paralle, get from upper line's "up","left-up"- the min dist
        if (sub_x == 0) // FIRST element speical, add DISTANCE here
            path_h2[blk_id] =  path_h1[blk_id] + dist[blk_id];
        else
            path_h2[blk_id] = min (path_h1[blk_id   ],
                                   path_h1[blk_id -1]);
        barrier(CLK_LOCAL_MEM_FENCE);
        // 2. serie, compare to left(front) element with myself, get the less value
        if (sub_x == 0) // the first element had plused dist, so can use here
            for (j=1; j<TRG_LEN; j++) 
                path_h2[j +blk_id] = min(path_h2[j   +blk_id], 
                                         path_h2[j-1 +blk_id])
                                   + dist[j +blk_id];
        barrier(CLK_LOCAL_MEM_FENCE);
        ex      = path_h2;
        path_h2 = path_h1;
        path_h1 = ex;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // when all done, can return the result
    if (sub_x == 0){
        RR[get_global_id(0) /TRG_LEN] = 
            sqrt(ex[get_local_id(0) +TRG_LEN -1]);
    }
}