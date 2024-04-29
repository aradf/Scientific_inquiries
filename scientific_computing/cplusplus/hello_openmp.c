
#include <stdio.h>     // printf
#include <stdlib.h>    //malloc, free
#include <omp.h>       // OpenMP
#include <math.h>      // pow
#include <unistd.h>    // sleep

/*
 * gcc -fopenmp -g -o ./run_compute hello_openmp.c -lm
 */

/*
 * OMP_NUM_THREADS is an environment variable.
 * export OMP_NUM_THREADS=8
 * 
 * False sharing occures when threads on different processor
 * modify variables that resdie on the same chche line.
 * 
 * Synchronization: Sync-ro-ni-zation h is scilent.
 * A section of code is executed by one thread at a time.
 * 
 * Atomic:
 * Update a single memory location/address
 * 
 * Barrier: Bar-ri-er
 * A barrier is a single point in the code where all active threads wait unitil all 
 * threads have arrived.
 * 
 * OpenMP Clauses
 * copyin:  Allows threads to access the master thread's value.
 * copyprivate: One or more variables should be shared among all threads.
 * default: behavior of unscoped variable.
 * firstprivate: each thread should have it's own instance of variable.
 * 
 */

int main(int argc, char** argv)
{
    /*
     * OpenMP (Open Multi-Processing) 
     * OpenMP parallel region.
     */
    
    omp_set_num_threads(4);
    #pragma omp parallel num_threads(4)
    {
        // open-mp function gets the name of each thread.
        int thread_id = omp_get_thread_num();
        int number_thread = omp_get_num_threads();

        if (thread_id == 0)
        {
            printf("id=0: There are %d threads\n", number_thread);
        }
        else
            {
                printf("id=%d: Hello World OpenMP ... \n",thread_id );
            }
    }
 
    // pointer variable: x == 0x1234, *x == 1,2,..n, &x == 0xABCD
    int * ptr = '\0';

    // allocate 100 integers.
    ptr = (int*) malloc (100 * sizeof(int));

    #pragma omp parallel num_threads(4)
    {
        // Open Multi process function gets the name of each thread.
        int i = omp_get_thread_num();
        int stride = 16;
        for (int k = 0; k < 2; k++)
        {
            // stride == 16 times the number of thread i.
            ptr[i*stride]++;
            printf("%d: ptr value %d \n", i, ptr[k]);
        }
    }
    
    // de-allcoate memory
    free (ptr);
    ptr = NULL;
    printf("main thread ... \n\n");

    double sum = 1.0;
    #pragma omp parallel num_threads(4)
    {
        // Open Multi process function gets the name of each thread.
        int id = omp_get_thread_num();
        #pragma omp critical
        sum += id*sum;
        printf("%d: sum = %f\n",id, sum);
    }
 
    printf("main thread ... \n\n");
    #pragma omp parallel num_threads(4)
    {
        // Open Multi process function gets the name of each thread.
        int id = omp_get_thread_num();
        #pragma omp atomic
        sum += id*sum;
        printf("%d: sum = %f\n",id, sum);
    }

    printf("main thread ... \n\n");
    #pragma omp parallel num_threads(4)
    {
        // Open Multi process function gets the name of each thread.
        int id = omp_get_thread_num();
        sum += id*sum;
        #pragma omp barrier
        printf("%d: sum = %f\n",id, sum);
    }

    printf("main thread ... \n\n");
    #pragma omp parallel num_threads(4)
    {
        // Open Multi process function gets the name of each thread.
        int id = omp_get_thread_num();
        sum += id*sum;
        #pragma omp single
        printf("%d: sum = %f\n",id, sum);
    }

    printf("main thread ... \n\n");
    {
        #pragma omp parallel for
        for (int iCnt = 0; iCnt < 10;iCnt++)
        {
            printf("Thread %d: i = %d\n",omp_get_thread_num(), iCnt);
        }
    }

    printf("main thread ... \n\n");
    {
        // col-lap-se 2 nested loops
        #pragma omp parallel for collapse(2)
        for (int x = -1; x <= 1 ; x+=1)
            for (int y = -1; y <= 1; y+=1)
            {
                printf("Thread %d: (%d, %d) \n",omp_get_thread_num(), x, y);
            }
    }

    printf("main thread ... \n\n");
    #define N 1000000
    double calc = 0.0;
    {
        // col-lap-se 2 nested loops
        #pragma omp parallel for reduction(+:calc)
        for (int iCnt = 0; iCnt < N; iCnt++)
            calc += pow(-1,iCnt) * 1.0/(2.0*iCnt + 1);
 
        printf("PI: %.12f \n", 4* calc);
    }

    // Scheduling - static
    printf("main thread ... \n\n");
    {
        #pragma omp parallel for schedule(static, 1) num_threads(4)
        for (int iCnt = 0; iCnt < 10; iCnt++)
        {
            // simulate with work load using sleep function.
            sleep(iCnt);
            printf("Thread %d Iteration %d: \n",omp_get_thread_num(), iCnt);
        }
    }

    // Scheduling - dynamic 
    printf("main thread ... \n\n");
    {
        #pragma omp parallel for schedule(dynamic, 1) num_threads(4)
        for (int iCnt = 0; iCnt < 10; iCnt++)
        {
            // simulate with work load using sleep function.
            sleep(iCnt);
            printf("Thread %d Iteration %d: \n",omp_get_thread_num(), iCnt);
        }
    }

    // data sharing
    int my_x = 0;
    printf("main thread ... \n\n");

    #pragma omp parallel num_threads(4)
    {
        int my_x = 5;
        printf("Thread %d Iteration %d: \n",omp_get_thread_num(), my_x);
    }

    #pragma omp parallel private(my_x) num_threads(4)
    {
        my_x = 1;
        printf("Thread %d inside private %d: \n",omp_get_thread_num(), my_x);
    }

    my_x = 15;
    #pragma omp parallel firstprivate(my_x) num_threads(4)
    {
        printf("Thread %d inside private %d: \n",omp_get_thread_num(), my_x);
    }

    {
        int my_x = 20;
        #pragma omp parallel for lastprivate(my_x) num_threads(4)
        for (int i = 0; i < 4; i++)
        {
            my_x = i;
            printf("Thread %d inside private %d: \n",omp_get_thread_num(), my_x);
        }
    }

    {
        int arr[1000], x = 10;
        #pragma omp parallel default(none) private(x) shared(arr)
        {
            x = 1; arr[0] = 2;
            printf("Inside x is %d and arr[0] is %d \n", x, arr[0]);
        }
        printf("outside x is %d and arr[0] is %d \n", x, arr[0]);
    }

    printf("main thread ... \n");
    return 0;
}