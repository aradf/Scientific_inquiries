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
   nvcc -G -g -o ./run_cuda matrix_multiply.cu
   nvcc -o ./run_cuda matrix_multiply.cu -lineinfo
   nvprof ./run_cuda
   (Not installed) nsys profile -t nvtx,cuda --stats=true --force-overwrite true --wait=all -o my_report ./run_cuda
   nv-nsight-cu-cli ./run_cuda
   nv-nsight-cu
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

__global__ void gpu_matrixMultiply(float *out,
                const float *first,
                const float *second,
                int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (column < n && row < n)
    {
        for (int i = 0; i < n; i++)
        {
            sum += first[row * n + i] * second[i * n + column];
        }
        out[row * n + column] = sum;
    }
}

// CPU  matrix_mult
void cpu_matrixMultiply(float *h_result, const float *h_a, const float *h_b, int n) 
{
    for (int i = 0; i < n; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            int tmp = 0.0;
            for (int k = 0; k < n; ++k) 
            {
                tmp += h_a[i * n + k] * h_b[k * n + j];
            }
            h_result[i * n + j] = tmp;
        }
    }
}

void init_matrix(float *a, float *b, int n)
{
    for (int cnt=0; cnt < n; cnt++)
    {
        a[cnt] = rand() % 1024;
    }

    for (int cnt=0; cnt < n; cnt++)
    {
        b[cnt] = rand() % 1024;
    }
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
    // 2^10 = 1024; Size of Matrix is 1024 * 1024
    int n = 1 << 10;    
    srand(3333);

    // size (in bytes) of matrix
    size_t bytes = n * n * sizeof(float);

    float * host_firstMatrix, * host_secondMatrix, * host_outputMatrix, *host_compareMatrix;
    float * device_firstMatrix, * device_secondMatrix, * device_outputMatrix;

    // Allocate host memory 
    host_firstMatrix = (float *)malloc(bytes);
    host_secondMatrix = (float *)malloc(bytes);
    host_outputMatrix = (float *)malloc(bytes);
    host_compareMatrix = (float *)malloc(bytes);

    init_matrix(host_firstMatrix, 
                host_secondMatrix, 
                n*n);
    // print_pointers(host_firstMatrix,
    //                n*n);

    // Allocate device memory for a
    cudaMalloc( (void**)&device_firstMatrix, bytes );
    cudaMalloc( (void**)&device_secondMatrix, bytes );
    cudaMalloc( (void**)&device_outputMatrix, bytes );

    // Transfer data from host to device memory
    cudaMemcpy(device_firstMatrix, host_firstMatrix, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_secondMatrix, host_secondMatrix, sizeof(float) * n, cudaMemcpyHostToDevice);

    //Cuda Launch.
    int block_size = 16;
    int grid_size = (int) ((n + block_size - 1) / block_size);

    // Use dim3 objects
    dim3 grid(grid_size,grid_size);
    dim3 threads(block_size,block_size);

    gpu_matrixMultiply<<< grid, threads>>>(device_outputMatrix, 
                                           device_firstMatrix, 
                                           device_secondMatrix, 
                                           n);
    cudaDeviceSynchronize();
    printf("Hello World From CPU!\n");

    cudaMemcpy(host_outputMatrix, device_outputMatrix, bytes, cudaMemcpyDeviceToHost);


    cpu_matrixMultiply(host_compareMatrix, host_firstMatrix, host_secondMatrix, n );

	// validate results computed by GPU
    bool all_elementsOk = true;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            
            if(host_compareMatrix[i*n + j] != host_outputMatrix[i*n + j])
            {
                all_elementsOk = false;
            }
        }
    }

    // roughly compute speedup
    if(all_elementsOk)
    {
        printf("All results are correct!!!");
    }
    else
    {
        printf("Incorrect results\n");
    }

    // print_pointers(host_outputMatrix, 
    //                n*n);

    free(host_firstMatrix);
    free(host_secondMatrix);
    free(host_outputMatrix);
    free(host_compareMatrix);

    // Cleanup after kernel execution
    cudaFree(device_firstMatrix);
    cudaFree(device_secondMatrix);
    cudaFree(device_outputMatrix);
    return 0;
}
