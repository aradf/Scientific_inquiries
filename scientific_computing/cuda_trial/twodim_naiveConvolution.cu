#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

// https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/05_convolution/1d_naive/convolution.cu

/*
   nvcc -G -g -o ./run_cuda twodim_naiveConvolution.cu
   nvcc -o ./run_cuda twodim_naiveConvolution.cu -lineinfo
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


// 7 x 7 convolutional mask
#define MASK_DIM 7

// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)

// Allocate mask in constant memory
__constant__ float constant_mask[MASK_DIM * MASK_DIM];

// 2D Convolution Kernel
// Takes:
//  matrix: Input matrix
//  result: Convolution result
//  N:      Dimensions of the matrices
__global__ void twodim_convollutionKernel(const float * input_matrix, 
                                          float * convolution_result, 
                                          int matrix_dimension) {

  // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Starting index for calculation
  int start_radius = row - MASK_OFFSET;
  int start_c = col - MASK_OFFSET;

  // Temp value for accumulating the result
  float temp = 0;

  // Iterate over all the rows
  for (int iCnt = 0; iCnt < MASK_DIM; iCnt++) 
  {
    // Go over each column
    for (int jCnt = 0; jCnt < MASK_DIM; jCnt++) 
    {
      // Range check for rows
      if ((start_radius + iCnt) >= 0 && (start_radius + iCnt) < matrix_dimension) 
      {
        // Range check for columns
        if ((start_c + jCnt) >= 0 && (start_c + jCnt) < matrix_dimension) 
        {
          // Accumulate result
          temp += input_matrix[(start_radius + iCnt) * matrix_dimension + (start_c + jCnt)] *
                          constant_mask[iCnt * MASK_DIM + jCnt];
        }
      }
    }
  }

  // Write back the result
  convolution_result[row * matrix_dimension + col] = temp;
}

/*
    Initializes an n x n matrix with random numbers
    Takes:
    input_matrix : Pointer to the matrix
    matrix_dimension : Dimension of the matrix (square)
*/
void init_matrix(float *input_matrix, int matrix_dimension) 
{
    for (int iCnt = 0; iCnt < matrix_dimension; iCnt++) 
    {
        for (int jCnt = 0; jCnt < matrix_dimension; jCnt++) 
        {
            input_matrix[matrix_dimension * iCnt + jCnt] = rand() % 100;
        }
    }
}

/*
    Verifies the 2D convolution result on the CPU
    Takes:
    m:      Original matrix
    mask:   Convolutional mask
    result: Result from the GPU
    N:      Dimensions of the matrix
*/
void verify_result(const float *original_matrix, 
                   const float *convolution_mask, 
                   float *result, 
                   int matrix_dimension) 
{
  // Temp value for accumulating results
  float temp;

  // Intermediate value for more readable code
  int offset_r;
  int offset_c;

  // Go over each row
  for (int iCnt = 0; iCnt < matrix_dimension; iCnt++) 
  {
    // Go over each column
    for (int jCnt = 0; jCnt < matrix_dimension; jCnt++) 
    {
        // Reset the temp variable
        temp = 0;

        // Go over each mask row
        for (int kCnt = 0; kCnt < MASK_DIM; kCnt++) 
        {
          // Update offset value for row
          offset_r = iCnt - MASK_OFFSET + kCnt;

          // Go over each mask column
            for (int lCnt = 0; lCnt < MASK_DIM; lCnt++) 
            {
              // Update offset value for column
              offset_c = jCnt - MASK_OFFSET + lCnt;

                // Range checks if we are hanging off the matrix
                if (offset_r >= 0 && offset_r < matrix_dimension) 
                {
                    if (offset_c >= 0 && offset_c < matrix_dimension) 
                    {
                      // Accumulate partial results
                      temp += original_matrix[offset_r * matrix_dimension + offset_c] * 
                                              convolution_mask[kCnt * MASK_DIM + lCnt];
                    }
                }
            }
        }
        // Fail if the results don't match
        assert(result[iCnt * matrix_dimension + jCnt] == temp);
    }
  }
}

int main() 
{
  // Dimensions of the matrix (2 ^ 10 x 2 ^ 10)
  int matrix_dimension = 1 << 10;

  // Size of the matrix (in bytes)
  size_t bytes_matrixSize = matrix_dimension * matrix_dimension * sizeof(float);

  // Allocate the matrix and initialize it
  float * cpu_matrix = new float[matrix_dimension * matrix_dimension];
  float * cpu_result = new float[matrix_dimension * matrix_dimension];

  init_matrix(cpu_matrix, 
              matrix_dimension);

  // Size of the mask in bytes
  size_t bytes_matrixMask = MASK_DIM * MASK_DIM * sizeof(float);

  // Allocate the mask and initialize it
  float * cpu_mask = new float[MASK_DIM * MASK_DIM];

  init_matrix(cpu_mask, 
              MASK_DIM);

  // Allocate device memory
  float * gpu_matrix;
  float * gpu_result;
  cudaMalloc(&gpu_matrix, 
             bytes_matrixSize);
  
  cudaMalloc(&gpu_result, 
             bytes_matrixSize);

  // Copy data to the device
  cudaMemcpy(gpu_matrix, 
             cpu_matrix, 
             bytes_matrixSize, 
             cudaMemcpyHostToDevice);
  
  cudaMemcpyToSymbol(constant_mask, 
                     cpu_mask, 
                     bytes_matrixMask);

  // Calculate grid dimensions
  int THREADS = 16;
  int BLOCKS = (matrix_dimension + THREADS - 1) / THREADS;

  // Dimension launch arguments
  dim3 block_dim(THREADS, 
                 THREADS);

  dim3 grid_dim(BLOCKS, 
                BLOCKS);

  // Perform 2D Convolution
  twodim_convollutionKernel<<<grid_dim, block_dim>>>(gpu_matrix, 
                                                     gpu_result, 
                                                     matrix_dimension);

  // Copy the result back to the CPU
  cudaMemcpy(cpu_result, 
             gpu_result, 
             bytes_matrixSize, 
             cudaMemcpyDeviceToHost);

  // Functional test
  verify_result(cpu_matrix, 
                cpu_mask, 
                cpu_result, 
                matrix_dimension);

  std::cout << "COMPLETED SUCCESSFULLY!" << std::endl;

  // Free the memory we allocated
  delete[] cpu_matrix;
  delete[] cpu_result;
  delete[] cpu_mask;

  cudaFree(gpu_matrix);
  cudaFree(gpu_result);

  return 0;
}
