#include <stdio.h>   
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
  Cuda C program offload computation to GPU.
  Cuda provides C/C++ language extension and
  API for programming and managing GPUs.

  The __global__ specifier indicates a cuda function that runs on 
  (GPU).  Such function is called "kernels" and it is a global 
  function.

 */

/*
   nvcc -G -g -o ./run_cuda vector_addUnified.cu
   nvcc -o ./run_cuda vector_add.cu -lineinfo
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

/*
 function returns a void, has a pointer variable of data type float.
 out has values like 0x1234, *out has values like 1.1 and &out has
 values like 0xABCD
 */
__global__ void vector_addUnified(float *out,
                float *first,
                float *second,
                int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
       out[tid] = first[tid] + second[tid];
        printf("tid: %d - %f at (%d %d)\n",tid, 
                                            out[tid], 
                                            blockIdx.x, 
                                            threadIdx.x);
    }

}

void print_pointers(float *out,
            int n)
{
   for (int cnt=0; cnt < n; cnt++)
   {
    printf("%f \n",out[cnt]);
   }
   printf("\n");
}

int ceiling(float float_number)
{
    int integer_value = trunc(float_number);
    return (integer_value + 1);
}

void check_answer(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
       assert(c[i] == a[i] + b[i]);
    }
}
int main()
{
    // Get the divce id for other Cuda Call.
    int id = cudaGetDevice( &id );
    
    // Declare the number of elements per array
    int n = 1 << 9;

    // Size of each array in bytes.
    size_t bytes = n * sizeof ( int );

    // Allocate host memory 
    float * out_array, * first_array, * second_array;

    // allocation of unified memory for these pointers.
    cudaMallocManaged( &out_array, 
                       bytes);
    cudaMallocManaged( &first_array, 
                       bytes);
    cudaMallocManaged( &second_array, 
                       bytes);

    for (int cnt = 0; cnt < n; cnt++)
    {
        first_array[cnt] = 1.1;
        second_array[cnt] = 2.2;
    }

    int BLOCK_SIZE = 256;
    int GRID_SIZE = (int)ceiling(n/BLOCK_SIZE);

    // pre-fetching the arrays to device (gpu)
    cudaMemPrefetchAsync(first_array, bytes, id);
    cudaMemPrefetchAsync(second_array, bytes, id);

    //Cuda Launch.
    vector_addUnified<<<GRID_SIZE, BLOCK_SIZE, 1>>>(out_array, 
                                                    first_array, 
                                                    second_array,  
                                                    n);
   
    // wait for all previous operations before using values
    cudaDeviceSynchronize();
    cudaMemPrefetchAsync(out_array, bytes, cudaCpuDeviceId);

    print_pointers(out_array, 
                   n);

    check_answer(first_array, second_array, out_array, n);

    return 0;
}
