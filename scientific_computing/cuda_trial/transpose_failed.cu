#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>

#define MAX_ERR 1e-6
#define TILE_DIM 4

/*
  Cuda C program offload computation to GPU.
  Cuda provides C/C++ language extension and
  API for programming and managing GPUs.

  The __global__ specifier indicates a cuda function that runs on 
  (GPU).  Such function is called "kernels" and it is a global 
  function.

 */

/*
   nvcc -G -g -o ./run_cuda transpose.cu
   nvcc -o ./run_cuda transpose.cu -lineinfo
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


// __global__ void vector_addGrid(float *out,
//                 float *first,
//                 float *second,
//                 int n)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid == 0)
//         printf("tid %d threadIdx.x %d blockIdx.x %d blockDim.x %d gridDim.x %d\n", 
//                 tid,   threadIdx.x,   blockIdx.x,   blockDim.x,   gridDim.x);
//     else
//         printf("tid %d threadIdx.x %d blockIdx.x %d\n", 
//                 tid,   threadIdx.x,   blockIdx.x   );
    
//     // Handling arbitrary vector size
//     out[tid] = first[tid] + second[tid];

// }


/*
 * Method transponse_onDevice runs on the device.  The pointer variable source has 
 * values like 0x1234, *source has values like 0.1f and 0xsource has values like 0xABCD
 * It is a pointer variable pointing to a memory block occuping a float data type 
 * 
*/
void transponse_onDevice(const float * source, 
                         float *destination, 
                         const unsigned int dimension)
{
    for (int yCnt = 0; yCnt < dimension; yCnt++)
    {
        for (int xCnt = 0; xCnt < dimension; xCnt++)
        {
            destination[yCnt + xCnt * dimension] = source[xCnt + yCnt * dimension];
        }
    }

}

__global__ void transpose_onGpu(const float *source,
                               float *distination,
                               const int dimension)
{
   __shared__ float tile[TILE_DIM][TILE_DIM + 1];
   int x_in = blockIdx.x * blockDim.x + threadIdx.x;
   int y_in = blockIdx.y * blockDim.y + threadIdx.y;

   int source_index = x_in + y_in * dimension ;
//    printf("Thread:%d: %f\n", source_index, source[source_index]);
//    printf("Thread:%d: %d %d\n", source_index, threadIdx.x, threadIdx.y);

   // Read from global memory to shared memory.  Global memory is aligned.
   tile[threadIdx.y][threadIdx.x] = source[source_index];

   // Wait for the threads in the block to finish, so the shared 
   // memory tile is filled.

   cooperative_groups::thread_block block =  cooperative_groups::this_thread_block();
   cooperative_groups::sync(block);

   // Output coordinates. Note that blockIdx.y is used to determine x_out
   // and blockIdx.x is used to determine y_out
   int x_out = blockIdx.y * blockDim.y + threadIdx.x;
   int y_out = blockIdx.x * blockDim.y + threadIdx.y;

   int destination_index = x_out + y_out * dimension;
   distination[destination_index] = tile[threadIdx.x][threadIdx.y];
   printf("Thread:%d: %f \n", source_index, distination[source_index]);

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

    float * device_distination, * device_source, * device_truthData;
    float * gpu_distination, * gpu_source;
    int n = 16;

    // Allocate host memory 
    device_distination = (float *)malloc(n * sizeof(float));
    device_source = (float *)malloc(n * sizeof(float));
    device_truthData = (float *)malloc(n * sizeof(float));
    
    for (int cnt = 0; cnt < n; cnt++)
    {
        device_source[cnt] = 0.0f + float(cnt);
        device_distination[cnt] = 0.0f;
        device_truthData[cnt] = 0.0f;
    }

    transponse_onDevice(device_source, device_truthData, n/4);
    printf("device_source!\n");
    print_pointers(device_source, n);
    printf("device_truthData!\n");
    print_pointers(device_truthData, n);

    // Allocate device memory for a
    cudaMalloc( (void**)&gpu_distination, sizeof(float) * n );
    cudaMalloc( (void**)&gpu_source, sizeof(float) * n );

    // Transfer data from host to device memory
    cudaMemcpy(gpu_source, device_source, sizeof(float) * n, cudaMemcpyHostToDevice);

    //Cuda Launch.
    int block_size = 256;
    // int grid_size = ((n + block_size) / block_size);

    transpose_onGpu<<< 1, 16>>>(gpu_source, 
                                gpu_distination, 
                                n/4);
    cudaDeviceSynchronize();

    cudaMemcpy(device_distination, gpu_distination, sizeof(float) * n, cudaMemcpyDeviceToHost);

    //verification
    // for (int i = 0 ; i < n; i++)
    // {
    //     assert(fabs(host_outPut[i] - host_firstArray[i] - host_secondArray[i]) < MAX_ERR);
    // }

    printf("device_distination!\n");
    print_pointers(device_distination, n);

    free(device_distination);
    free(device_source);
    free(device_truthData);

    // Cleanup after kernel execution
    cudaFree(gpu_distination);
    cudaFree(gpu_source);
    return 0;
}
