#include <stdio.h>

/*
  Cuda C program offload computation to GPU.
  Cuda provides C/C++ language extension and
  API for programming and managing GPUs.

  The __global__ specifier indicates a cuda function that runs on 
  (GPU).  Such function is called "kernels" and it is a global 
  function.

 */

/*
   nvcc -o ./run_cuda hello.cu

  'lsb release -r' 
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

 */

__global__ void cuda_hello()
{
    printf("Hello World From GPU!\n");
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
        printf("tid %d threadIdx.x %d blockIdx.x %d blockDim.x %d gridDim.x %d\n", 
                tid,   threadIdx.x,   blockIdx.x,   blockDim.x,   gridDim.x);
    else
        printf("tid %d threadIdx.x %d blockIdx.x %d\n", 
                tid,   threadIdx.x,   blockIdx.x   );
}

int main()
{
    //Cuda Launch.
    cuda_hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("Hello World From CPU!\n");
    return 0;
}
