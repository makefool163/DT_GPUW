A python module with GPU acceleration dtw algorithm

1. About it
 
  This a CPU speed module for python.
  It has two special : CUDA and OpenCL.

2. How to Use
 
  2.1 Install development kit
  
  In first, you must install the development kit for your graphics cards.  
  For my test, AMD's don't need development kit, and NVIDIA's need install the CUDA kit.
  
  2.2 Python Module
  
  CUDA dtw need you install pycuda module first, the example in cudadtw.py tell you how to use it.
  OpenCL dtw need you install pyopencl module first, the example in opencldtw.py tell you how to use it.
  
3. Special
  
  The openCL module is slightly faster than cuda's, even on NVIDIA graphics cards.
  And openCL module is not faster than the CPU's in Intel IGP.