#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_ERR 1e-6
/*
  Cuda C program offload computation to GPU.
  Cuda provides C/C++ language extension and
  API for programming and managing GPUs.

  The __global__ specifier indicates a cuda function that runs on 
  (GPU).  Such function is called "kernels" and it is a global 
  function.

 */

/*
   nvcc -G -g -o ./run_cuda vector_addGrid.cu
   nvcc -o ./run_cuda vector_addGrid.cu -lineinfo
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

/**
 * The threadIdx.x is the index of the thread within a block.
 * The blockDim.x is the size of the thread block (Number of threads in a thread block)
 * vector_add<<<1, 256>>>(cuda_outPut, cuda_firstArray, cuda_secondArray, n);
 * The threadIdx ranges from 0 to 255
 * The value of the blockDim.x is 256
 * 
 */


__global__ void vector_addGrid(float *out,
                float *first,
                float *second,
                int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
        printf("tid %d threadIdx.x %d blockIdx.x %d blockDim.x %d gridDim.x %d\n", 
                tid,   threadIdx.x,   blockIdx.x,   blockDim.x,   gridDim.x);
    else
        printf("tid %d threadIdx.x %d blockIdx.x %d\n", 
                tid,   threadIdx.x,   blockIdx.x   );
    
    // Handling arbitrary vector size
    out[tid] = first[tid] + second[tid];

}


void print_pointers(float *out,
            int n)
{
   for (int cnt=0; cnt < n; cnt++)
   {
    printf("%f\n",out[cnt]);
   }
   printf("\n");
}

int main()
{

    float * host_firstArray, * host_secondArray, * host_outPut;
    float * cuda_firstArray, * cuda_secondArray, * cuda_outPut;
    int n = 2*256;

    // Allocate host memory 
    host_firstArray = (float *)malloc(n * sizeof(float));
    host_secondArray = (float *)malloc(n * sizeof(float));
    host_outPut = (float *)malloc(n * sizeof(float));
    
    for (int cnt = 0; cnt < n; cnt++)
    {
        host_firstArray[cnt] = 1.1f;
        host_secondArray[cnt] = 2.2f;
    }

    // Allocate device memory for a
    cudaMalloc( (void**)&cuda_firstArray, sizeof(float) * n );
    cudaMalloc( (void**)&cuda_secondArray, sizeof(float) * n );
    cudaMalloc( (void**)&cuda_outPut, sizeof(float) * n );

    // Transfer data from host to device memory
    cudaMemcpy(cuda_firstArray, host_firstArray, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_secondArray, host_secondArray, sizeof(float) * n, cudaMemcpyHostToDevice);

    //Cuda Launch.
    int block_size = 256;
    int grid_size = ((n + block_size) / block_size);

    vector_addGrid<<< (grid_size - 1), block_size>>>(cuda_outPut, 
                                    cuda_firstArray, 
                                    cuda_secondArray, 
                                    n);
    cudaDeviceSynchronize();
    printf("Hello World From CPU!\n");

    cudaMemcpy(host_outPut, cuda_outPut, sizeof(float) * n, cudaMemcpyDeviceToHost);

    //verification
    for (int i = 0 ; i < n; i++)
    {
        assert(fabs(host_outPut[i] - host_firstArray[i] - host_secondArray[i]) < MAX_ERR);
    }

    print_pointers(host_outPut, n);

    free(host_firstArray);
    free(host_secondArray);
    free(host_outPut);

    // Cleanup after kernel execution
    cudaFree(cuda_firstArray);
    cudaFree(cuda_secondArray);
    cudaFree(cuda_outPut);
    return 0;
}
