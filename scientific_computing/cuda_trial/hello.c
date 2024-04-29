#include <stdio.h>

/*
 gcc -o ./run_c ./hello.c
 cd /usr/include/
*/

void c_hello()
{
    printf("Hello world!\n");
}

int main()
{
    c_hello();
    return 0;
}