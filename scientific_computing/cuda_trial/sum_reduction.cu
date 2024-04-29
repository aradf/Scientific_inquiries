// This program performs sum reduction with an optimization
// removing warp divergence
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
// using std::cout;
// using std::generate;
using std::vector;

#define SHMEM_SIZE 256

__global__ void sumReduction(float *v, float *v_r) 
{
  // Allocate shared memory
  __shared__ int partial_sum[SHMEM_SIZE];

  // Calculate thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Load elements into shared memory
  partial_sum[threadIdx.x] = v[tid];
  __syncthreads();

  // Increase the stride of the access until we exceed the CTA dimensions
  for (int s = 1; s < blockDim.x; s *= 2) 
  {
    // Change the indexing to be sequential threads
    int index = 2 * s * threadIdx.x;

    // Each thread does work unless the index goes off the block
    if (index < blockDim.x) 
    {
      partial_sum[index] += partial_sum[index + s];
    }
    __syncthreads();
  }

  // Let the thread 0 for this block write it's result to main memory
  // Result is inexed by this block
  if (threadIdx.x == 0) 
  {
    v_r[blockIdx.x] = partial_sum[0];
  }
}

void print_vector( vector<float> &some_vector )
{
   for (int iCnt = 0; iCnt < some_vector.size(); iCnt++)
   {
      std::cout << some_vector[iCnt] << " " ;
   }
   std::cout << std::endl;
}

void  copy_vectorPointer(vector<float> &some_vector, float * some_pointer)
{
   for (int iCnt = 0; iCnt <some_vector.size(); iCnt++)
   {
      some_pointer[iCnt] = some_vector[iCnt];
   }
   
}

void print_pointer(float * some_pointer, int n)
{
   for (int iCnt = 0; iCnt <n; iCnt++)
   {
      std::cout << some_pointer[iCnt] << " ";
   }
   std::cout << std::endl;
}

/*
    CUDA Basic Linear Algerbra Subroutine Library (CuBLAS)
    nvcc -o ./run_cuda -lineinfo -G -g sum_reduction.cu
    nvcc -o ./run_cuda -G -g sum_reduction.cu

    nvprof ./run_cuda
    (Not installed) nsys profile -t nvtx,cuda --stats=true --force-overwrite true --wait=all -o my_report ./run_cuda
    nv-nsight-cu-cli ./run_cuda
    nv-nsight-cu
    cuda-memcheck ./run_cuda
    compute-sanitizer ./run_cuda
    nvvp ./run_cuda          # must build with -lineinfo option.
*/


int main() 
{
  // Vector size
  int N = 1 << 16;
  size_t bytes = N * sizeof(float);
  float * cpu_vectorPtr = nullptr;
  
  cpu_vectorPtr = (float *)malloc (bytes);

  // Host data
  vector<float> cpu_vector(N);
  vector<float> cpu_vectorReduction(N);

  // Initialize the input data
  std::generate(begin(cpu_vector), 
                end(cpu_vector), 
                []() { return rand() % 10; });

  print_vector(cpu_vector);
  copy_vectorPointer(cpu_vector, 
                     cpu_vectorPtr);

  print_pointer(cpu_vectorPtr, N);

  // Allocate device memory
  float *gpu_vector = nullptr, *gpu_vectorReduction = nullptr;

  cudaMalloc(&gpu_vector, 
              bytes);

  cudaMalloc(&gpu_vectorReduction, 
              bytes);

  // Copy to device
//   cudaMemcpy(gpu_vector, 
//              cpu_vector.data(), 
//              bytes, 
//              cudaMemcpyHostToDevice);

  cudaMemcpy(gpu_vector, 
             cpu_vectorPtr, 
             bytes, 
             cudaMemcpyHostToDevice);

  // TB Size
  const int TB_SIZE = 256;

  // Grid Size (No padding)
  int GRID_SIZE = N / TB_SIZE;

  // Call kernels
  sumReduction<<<GRID_SIZE, TB_SIZE>>>(gpu_vector, gpu_vectorReduction);

  sumReduction<<<1, TB_SIZE>>>(gpu_vectorReduction, gpu_vectorReduction);

  // Copy to host;
  cudaMemcpy(cpu_vectorReduction.data(), 
             gpu_vectorReduction, 
             bytes, 
             cudaMemcpyDeviceToHost);

  // Print the result
  assert(cpu_vectorReduction[0] == std::accumulate(begin(cpu_vector), 
                                                   end(cpu_vector), 
                                                   0));

  free(cpu_vectorPtr);
  cpu_vectorPtr = nullptr;
  
  cudaFree(gpu_vector);
  cudaFree(gpu_vectorReduction);

  gpu_vector = nullptr;
  gpu_vectorReduction = nullptr;

  std::cout << "COMPLETED SUCCESSFULLY\n";
  return 0;
}