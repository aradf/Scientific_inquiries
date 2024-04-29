#include <stdio.h>
#include <stdlib.h>

/*
 gcc -g -o ./run_c ./vector_add.c
 -g: Debug
 -o: output file.
 cd /usr/include/
*/

 /*
  CPU memory is called host memory and GPU memory is called device memory.
  Pointers variables to the CPU memory block are called host pointer and 
  pointers variables to the GPU memory block are called device pointers.

  */
 

#define N 10000000

/*
 function returns a void, has a pointer variable of data type float.
 out has values like 0x1234, *out has values like 1.1 and &out has
 values like 0xABCD
 */
void vector_add(float *out,
                float *first,
                float *second,
                int n)
{
    for (int cnt=0; cnt < n; cnt++)
    {
        out[cnt] = first[cnt] + second[cnt];
    }
}

void print_pointers(float *out,
            int n)
{
   for (int cnt=0; cnt < n; cnt++)
   {
    printf("%f ",out[cnt]);
   }
   printf("\n");
}


int main()
{
    float * first_array;
    float * second_array;
    float * out_put;
    int n = 8;

    first_array = (float *)malloc(8 * sizeof(float));
    second_array = (float *)malloc(8 * sizeof(float));
    out_put = (float *)malloc(8 * sizeof(float));

    for (int cnt = 0; cnt < n; cnt++)
    {
        first_array[cnt] = 1.1;
        second_array[cnt] = 2.2;
    }

    print_pointers(first_array, n);
    print_pointers(second_array, n);

    vector_add(out_put, first_array, second_array, n);
    print_pointers(out_put, n);

    free(first_array);
    free(second_array);
    free(out_put);

    return 0;
}