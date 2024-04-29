#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

// https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/05_convolution/1d_naive/convolution.cu

/*
   nvcc -G -g -o ./run_cuda onedim_naiveConvolution.cu
   nvcc -o ./run_cuda naiveConvolution.cu -lineinfo
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

/*
  One dimentional convolution kernel
  Arguments:
    array  = padded_array,
    mask   = convolution mask,
    result = result array,
    n      = number of elements in array.
    m      = number of elements in mask.
*/
__global__ void onedim_convolutionKernel(const float * input_array, 
                                         const float * input_mask, 
                                         float * output_result, 
                                         int number_arrayElements, 
                                         int number_maskElements)
{
   // Global Thread id calculation.
   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   // calcualte radius of mask
   int radius = number_maskElements / 2;

   // calculate the starting point of the element
   int start = tid - radius;

   float temp = 0.0;

   // traverse over each element of the mask
   for (int jCnt = 0; jCnt < number_maskElements; jCnt++)
   {
      // ignore element that are off
      if ((start + jCnt >= 0) && (start + jCnt < number_arrayElements))
      {
         // accumualte partial results
         temp += input_array[start + jCnt] * input_mask[jCnt];
      }
   }
   output_result[tid] = temp;
}

void verify_onCPU(const float * input_array, const float * input_mask, const float * output_result, int number_arrayElements, int number_maskElements)
{
    int radius = number_maskElements / 2;
    int temp;
    int start;
    for (int iCnt = 0; iCnt < number_arrayElements; iCnt++) 
    {
        start = iCnt - radius;
        temp = 0.0;
        for (int jCnt = 0; jCnt < number_maskElements; jCnt++) 
        {
            if ((start + jCnt >= 0) && (start + jCnt < number_arrayElements)) 
            {
                temp += input_array[start + jCnt] * input_mask[jCnt];
            }
        }
        assert(temp == output_result[iCnt]);
    }
}

int main()
{
    // Number of element sin the result array
    int n = 1 << 20;

    // size of array in bytes
    int array_byteSize = n * sizeof(float);

    // Number of elements in the convolution mask
    int m = 7;

    // Size of mask in bytes
    int mask_byteSize = m * sizeof(int);

    // Allocate the array (include edge elements)...
    std::vector<float> cpu_array(n);

    // ... and initialize it
    std::generate(begin(cpu_array), 
                  end(cpu_array), 
                  [](){ return rand() % 100; });

    // Allocate the mask and initialize it
    std::vector<float> cpu_mask(m);
    std::generate(begin(cpu_mask), 
                  end(cpu_mask), 
                  [](){ return rand() % 10; });

    // Allocate space for the result
    std::vector<float> cpu_result(n);

    // Allocate space on the device
    float *gpu_array, *gpu_mask, *gpu_result;
    cudaMalloc(&gpu_array, array_byteSize);
    cudaMalloc(&gpu_mask, mask_byteSize);
    cudaMalloc(&gpu_result, array_byteSize);

    // Copy the data to the device
    cudaMemcpy(gpu_array, cpu_array.data(), array_byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mask, cpu_mask.data(), mask_byteSize, cudaMemcpyHostToDevice);

    // Threads per TB
    int THREADS = 256;

    // Number of TBs
    int GRID = (n + THREADS - 1) / THREADS;

    // Call the kernel
    onedim_convolutionKernel<<<GRID, THREADS>>>(gpu_array, gpu_mask, gpu_result, n, m);

    // Copy back the result
    cudaMemcpy(cpu_result.data(), gpu_result, array_byteSize, cudaMemcpyDeviceToHost);
 
    // Verify the result
    verify_onCPU(cpu_array.data(), cpu_mask.data(), cpu_result.data(), n, m);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    // Free allocated memory on the device and host
    cudaFree(gpu_result);
    cudaFree(gpu_mask);
    cudaFree(gpu_array);

    return EXIT_SUCCESS;
}