#include "stdio.h"
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

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}
// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the matrix multiplication kernel
__global__ void matrix_multiKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void matrix_multiGpuWrapper(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    matrix_multiKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
// Matrix multiplication kernel called by matrix_multiGpuWrapper()
 __global__ void matrix_multiKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) 
    {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

void matrix_multCpu(const Matrix A, const Matrix B, Matrix C) 
{
   float sum = 0.f;

   for (int row = 0; row < A.width; row++ )
   {
      for (int col = 0; col < A.height; col++)
      {
        sum = 0.0;
        for (int stride = 0; stride < A.height; stride++)
        {
            sum += A.elements[row * (A.height) + stride] * B.elements[stride * (A.height) + col];
        }
        C.elements[row * (A.height) + col] = sum;
      }
   }

}

void compare_pointers(const Matrix A, const Matrix B)
{
   for (int cnt=0; cnt < A.width * A.height; cnt++)
   {
    printf("%f - %f\n",A.elements[cnt], B.elements[cnt]);
   }
}

int main(int argc, char ** argv)
{
    // Matrices for the program
    Matrix  cpu_firstMatrix;
    Matrix  cpu_secondMatrix;
    Matrix  cpu_referenceMatrix;
    Matrix  gpu_referenceMatrix;

    // const int MATRIX_SIZE = 1024 - 1 << 10;
    // const int MATRIX_SIZE = 256 - 1 << 8
    // const int MATRIX_SIZE = 64 - 1 << 6
    // const int MATRIX_SIZE = 32 - 1 << 5
    const int MATRIX_SIZE = 1 << 8;
    int size = MATRIX_SIZE * MATRIX_SIZE;
    srand(2012);

    cpu_firstMatrix.height = MATRIX_SIZE;
    cpu_firstMatrix.width = MATRIX_SIZE;
    cpu_firstMatrix.elements = NULL;

    cpu_secondMatrix.height = MATRIX_SIZE;
    cpu_secondMatrix.width = MATRIX_SIZE;
    cpu_secondMatrix.elements = NULL;

    cpu_referenceMatrix.height = MATRIX_SIZE;
    cpu_referenceMatrix.width = MATRIX_SIZE;
    cpu_referenceMatrix.elements = NULL;

    gpu_referenceMatrix.height = MATRIX_SIZE;
    gpu_referenceMatrix.width = MATRIX_SIZE;
    gpu_referenceMatrix.elements = NULL;

    cpu_firstMatrix.elements = (float *) malloc (size * ( sizeof(float)));
    cpu_secondMatrix.elements = (float *) malloc (size * ( sizeof(float)));
    cpu_referenceMatrix.elements = (float *) malloc (size * ( sizeof(float)));
    gpu_referenceMatrix.elements = (float *) malloc (size * ( sizeof(float)));

    for (int i = 0; i < size; i++)
    {
        cpu_firstMatrix.elements[i] = rand() / (float)RAND_MAX;
        cpu_secondMatrix.elements[i] = rand() / (float)RAND_MAX;
    }

    matrix_multCpu(cpu_firstMatrix, 
                      cpu_secondMatrix, 
                      cpu_referenceMatrix);

    matrix_multiGpuWrapper(cpu_firstMatrix, 
                      cpu_secondMatrix, 
                      gpu_referenceMatrix);

    // print_pointers(cpu_referenceMatrix);
    // print_pointers(gpu_referenceMatrix);
    compare_pointers(cpu_referenceMatrix, gpu_referenceMatrix);

    free(cpu_firstMatrix.elements);
    cpu_firstMatrix.elements = NULL;
    
    free(cpu_secondMatrix.elements);
    cpu_secondMatrix.elements = NULL;

    free(cpu_referenceMatrix.elements);
    cpu_referenceMatrix.elements = NULL;

    free(gpu_referenceMatrix.elements);
    gpu_referenceMatrix.elements = NULL;

    return 0;
}
