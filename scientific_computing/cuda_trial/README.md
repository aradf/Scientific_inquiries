

https://www.youtube.com/watch?v=3yqpirzxudw&t=23s
https://www.youtube.com/watch?v=fVJ0WMCRy_k&t=253s
https://www.youtube.com/watch?v=12iWT-6viow&list=PLuuY9GP0b7ZyAvApKjY9rsU5vsnO4NtRu


/*
   Cuda C program offload computation to GPU.
   Cuda provides C/C++ language extension and
   API for programming and managing GPUs.

   The __global__ specifier indicates a cuda function that runs on 
   (GPU).  Such function is called "kernels" and it is a global 
   function.

   CUDA Basic Linear Algerbra Subroutine Library (CuBLAS)
   nvcc -o ./run_cuda -lineinfo -G -g -lcublas vector_addcuBLAS.cu
   nvcc -o ./run_cuda -G -g -lcublas vector_addcuBLAS.cu

   nvprof ./run_cuda
   (Not installed) nsys profile -t nvtx,cuda --stats=true --force-overwrite true --wait=all -o my_report ./run_cuda
   nv-nsight-cu-cli ./run_cuda
   nv-nsight-cu
   cuda-memcheck ./run_cuda
   compute-sanitizer ./run_cuda
   nvvp ./run_cuda          # must build with -lineinfo option.


 */



/*
   nvcc -G -g -o ./run_cuda matrix_multiply3.cu
   nvcc -o ./run_cuda matrix_multiply3.cu -lineinfo
   nvprof ./run_cuda
   cuda-memcheck ./run_cuda
   compute-sanitizer ./run_cuda
   nvvp ./run_cuda          # must build with -lineinfo option.

  'lsb releae -r' 
        provides Ubuntu 20.04
  'nvidia-smi'
  'nvidia-smi --query-gpu=name --format=csv,noheader'
        NVIDIA GeForce RTX 3060 Laptop GPU
  
  sudo apt install nvidia-cuda-toolkit
  which nvcc
  cuobjdump ./run_cuda

  /user/include
  /user/include/c++/9
  /user/include/cuda-gdb

  CUDA organizes threads in groups named "threadblock" and the kernel can 
  launch multiple thread blocks, organized into a "grid" structuer.

  <<< M, T >>> which means grid of M thread blocks and each thread block has T
  parallel threads.

  CUDA GPU has multiple parallel processor calling Streaming Multiprocessor or SMs.
  Using multiple thread blocks is excersized here.
  blockIdx.x contains the index of the lbock within a Grid.
  gridDim.x contains the size of the grid.

 */

