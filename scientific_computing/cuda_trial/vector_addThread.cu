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
   nvcc -G -g -o ./run_cuda vector_addThread.cu
   nvcc -o ./run_cuda vector_addThread.cu -lineinfo
   nvprof ./run_cuda
   cuda-memcheck ./run_cuda
   compute-sanitizer ./run_cuda

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

 */

/**
 * The threadIdx.x is the index of the thread within a block.
 * The blockDim.x is the size of the thread block (Number of threads in a thread block)
 * vector_add<<<1, 256>>>(cuda_outPut, cuda_firstArray, cuda_secondArray, n);
 * The threadIdx ranges from 0 to 255
 * The value of the blockDim.x is 256
 * 
 */


__global__ void vector_addThread(float *out,
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

    out[tid]  = first[tid] + second[tid];
}

__global__ void vector_addThreadTest(float *out,
                float *first,
                float *second,
                int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int cnt=index; cnt < n; cnt += stride)
    {
        out[cnt] = first[cnt] + second[cnt];
        printf("Inside Loop: tid %d threadIdx.x %d blockIdx.x %d index %d stride %d\n", 
                             tid,   threadIdx.x,   blockIdx.x,   index,   stride);
    }
    if (tid == 0)
        printf("tid %d threadIdx.x %d blockIdx.x %d blockDim.x %d gridDim.x %d index %d stride %d\n", 
                tid,   threadIdx.x,   blockIdx.x,   blockDim.x,   gridDim.x,   index,   stride);

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
    int n = 1*256;

    // Allocate host memory 
    host_firstArray = (float *)malloc(n * sizeof(float));
    host_secondArray = (float *)malloc(n * sizeof(float));
    host_outPut = (float *)malloc(n * sizeof(float));
    

    // Allocate device memory for a
    cudaMalloc( (void**)&cuda_firstArray, sizeof(float) * n );
    cudaMalloc( (void**)&cuda_secondArray, sizeof(float) * n );
    cudaMalloc( (void**)&cuda_outPut, sizeof(float) * n );

    for (int cnt = 0; cnt < n; cnt++)
    {
        host_firstArray[cnt] = 1.1;
        host_secondArray[cnt] = 2.2;
    }

    // Transfer data from host to device memory
    cudaMemcpy(cuda_firstArray, host_firstArray, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_secondArray, host_secondArray, sizeof(float) * n, cudaMemcpyHostToDevice);

    //Cuda Launch.
    // vector_addThread<<<1, 256>>>(cuda_outPut, 
    //                                 cuda_firstArray, 
    //                                 cuda_secondArray, 
    //                                 n);

    vector_addThreadTest<<<1, 256>>>(cuda_outPut, 
                                    cuda_firstArray, 
                                    cuda_secondArray, 
                                    n);

    cudaDeviceSynchronize();
    printf("Hello World From CPU!\n");

    cudaMemcpy(host_outPut, cuda_outPut, sizeof(float) * n, cudaMemcpyDeviceToHost);

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
