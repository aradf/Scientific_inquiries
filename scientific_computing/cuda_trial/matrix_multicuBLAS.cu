//Example 1. Application Using C and cuBLAS: 1-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <assert.h>

void print_pointers(const float * A, int n)
{
   for (int cnt=0; cnt < n; cnt++)
   {
    printf("%d: %f\n",cnt, A[cnt]);
   }
}

/*
    CUDA Basic Linear Algerbra Subroutine Library (CuBLAS)
    nvcc -o ./run_cuda -lineinfo -G -g -lcublas -lcurand matrix_multicuBLAS.cu
    nvcc -o ./run_cuda -G -g -lcublas -lcurand matrix_multicuBLAS.cu

    nvprof ./run_cuda
    (Not installed) nsys profile -t nvtx,cuda --stats=true --force-overwrite true --wait=all -o my_report ./run_cuda
    nv-nsight-cu-cli ./run_cuda
    nv-nsight-cu
    cuda-memcheck ./run_cuda
    compute-sanitizer ./run_cuda
    nvvp ./run_cuda          # must build with -lineinfo option.
*/

/*
   Row major order (row is contingous in memory) the elements of a row are next to each other in memory.
   Column major order (column is contingous in memory) the elments of a column are next to eac other in memory.
 */

int main (void)
{
    int n = 1 << 10;
    size_t bytes = n * n * sizeof(float);
    
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle=nullptr;

    float* gpu_ptrA = nullptr;
    float* gpu_ptrB = nullptr;
    float* gpu_ptrC = nullptr;

    float* cpu_ptrA = nullptr;
    float* cpu_ptrB = nullptr;
    float* cpu_ptrC = nullptr;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) 
    {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    cpu_ptrA = (float *)malloc (bytes);
    if (!cpu_ptrA) 
    {
        printf ("host memory (cpu) allocation failed");
        return EXIT_FAILURE;
    }

    cpu_ptrB = (float *)malloc (bytes);
    if (!cpu_ptrB) 
    {
        printf ("host memory (cpu) allocation failed");
        return EXIT_FAILURE;
    }

    cpu_ptrC = (float *)malloc (bytes);
    if (!cpu_ptrC) 
    {
        printf ("host memory (cpu) allocation failed");
        return EXIT_FAILURE;
    }


    cudaStat = cudaMalloc ((void**)&gpu_ptrA, bytes);
    if (cudaStat != cudaSuccess) 
    {
        printf ("device memory (gpu) allocation failed");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc ((void**)&gpu_ptrB, bytes);
    if (cudaStat != cudaSuccess) 
    {
        printf ("device memory (gpu) allocation failed");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc ((void**)&gpu_ptrC, bytes);
    if (cudaStat != cudaSuccess) 
    {
        printf ("device memory (gpu) allocation failed");
        return EXIT_FAILURE;
    }

    // Pseduo random number generator
    curandGenerator_t random_numberGenerator;
    curandCreateGenerator(&random_numberGenerator, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed
    curandSetPseudoRandomGeneratorSeed(random_numberGenerator, (unsigned long long)clock());

    // Fill the matrix with random numbers on the gpu/device
    curandGenerateUniform(random_numberGenerator, gpu_ptrA, n * n);
    curandGenerateUniform(random_numberGenerator, gpu_ptrB, n * n);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, 
                CUBLAS_OP_N, 
                CUBLAS_OP_N, 
                n, 
                n, 
                n, 
                &alpha, 
                gpu_ptrA, 
                n, 
                gpu_ptrB, 
                n, 
                &beta, 
                gpu_ptrC, 
                n);

    cudaMemcpy(cpu_ptrA, gpu_ptrA, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_ptrB, gpu_ptrB, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_ptrC, gpu_ptrC, bytes, cudaMemcpyDeviceToHost);

    cudaFree (gpu_ptrA);
    cudaFree (gpu_ptrB);
    cudaFree (gpu_ptrC);
 
    gpu_ptrA = nullptr;
    gpu_ptrB = nullptr;    
    gpu_ptrC = nullptr;
    cublasDestroy(handle);
    
    free(cpu_ptrA);
    free(cpu_ptrB);
    free(cpu_ptrC);

    cpu_ptrA = nullptr;
    cpu_ptrB = nullptr;    
    cpu_ptrC = nullptr;

    return EXIT_SUCCESS;

}