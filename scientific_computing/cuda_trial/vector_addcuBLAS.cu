//Example 1. Application Using C and cuBLAS: 1-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define M 6
#define N 5
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

void print_pointers(const float * A, int n)
{
   for (int cnt=0; cnt < n; cnt++)
   {
    printf("%d: %f\n",cnt, A[cnt]);
   }
}

/*
    CUDA Basic Linear Algerbra Subroutine Library (CuBLAS)
    nvcc -o ./run_cuda -lineinfo -G -g -lcublas vector_addcuBLAS.cu
    nvcc -o ./run_cuda -G -g -lcublas vector_addcuBLAS.cu

    nvprof ./run_cuda
    (Not installed) nsys profile -t nvtx,cuda --stats=true --force-overwrite true --wait=all -o my_report ./run_cuda
    nv-nsight-cu-cli ./run_cuda
    nv-nsight-cu
    cuda-memcheck ./run_cuda
    compute-sanitizer ./run_cuda
    nvvp ./run_cuda          # must build with -lineinfo option.
*/

int main (void)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle=nullptr;

    int iCnt = 0, jCnt = 0;
    float* gpu_ptrA = nullptr;
    float* gpu_ptrB = nullptr;
    float* cpu_ptrA = nullptr;
    float* cpu_ptrB = nullptr;
    float* cpu_ptrC = nullptr;

    stat = cublasCreate(&handle);
    
    if (stat != CUBLAS_STATUS_SUCCESS) 
    {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    cpu_ptrA = (float *)malloc (M * N * sizeof (*cpu_ptrA));

    if (!cpu_ptrA) 
    {
        printf ("host memory (cpu) allocation failed");
        return EXIT_FAILURE;
    }

    cpu_ptrB = (float *)malloc (M * N * sizeof (*cpu_ptrB));
    if (!cpu_ptrB) 
    {
        printf ("host memory (cpu) allocation failed");
        return EXIT_FAILURE;
    }

    cpu_ptrC = (float *)malloc (M * N * sizeof (*cpu_ptrC));
    if (!cpu_ptrC) 
    {
        printf ("host memory (cpu) allocation failed");
        return EXIT_FAILURE;
    }

    // populate the host pointer
    for (jCnt = 1; jCnt <= N; jCnt++) 
    {
        for (iCnt = 1; iCnt <= M; iCnt++) 
        {
            cpu_ptrA[IDX2F(iCnt,jCnt,M)] = (float)((iCnt-1) * N + jCnt);
            cpu_ptrB[IDX2F(iCnt,jCnt,M)] = (float)((iCnt-1) * N + jCnt);
        }
    }
    
    print_pointers(cpu_ptrA, (N*M));

    cudaStat = cudaMalloc ((void**)&gpu_ptrA, M*N*sizeof(*cpu_ptrA));
    if (cudaStat != cudaSuccess) 
    {
        printf ("device memory (gpu) allocation failed");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc ((void**)&gpu_ptrB, M*N*sizeof(*cpu_ptrB));
    if (cudaStat != cudaSuccess) 
    {
        printf ("device memory (gpu) allocation failed");
        return EXIT_FAILURE;
    }

    cublasSetVector(N*M, sizeof(float), cpu_ptrA, 1, gpu_ptrA, 1);
    cublasSetVector(N*M, sizeof(float), cpu_ptrB, 1, gpu_ptrB, 1);

    // Launch smple saxpy kernel (single precision a * x + y)
    const float scale = 1.0f;
    cublasSaxpy(handle, N*M, &scale, gpu_ptrA, 1, gpu_ptrB, 1);

    cublasGetVector(N*M, sizeof(float), gpu_ptrB, 1, cpu_ptrC, 1);
    
    cudaFree (gpu_ptrA);
    cudaFree (gpu_ptrB);
    cublasDestroy(handle);
    for (jCnt = 1; jCnt <= N; jCnt++) 
    {
        for (iCnt = 1; iCnt <= M; iCnt++) {
            printf ("%d: %f %f \n",iCnt, cpu_ptrA[IDX2F(iCnt,jCnt,M)], cpu_ptrC[IDX2F(iCnt,jCnt,M)]);
        }
    }
    
    free(cpu_ptrA);
    free(cpu_ptrB);
    free(cpu_ptrC);

    cpu_ptrA = nullptr;
    cpu_ptrB = nullptr;    
    cpu_ptrC = nullptr;

    return EXIT_SUCCESS;

}