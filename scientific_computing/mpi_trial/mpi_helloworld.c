#include <mpi.h>
#include <stdio.h>

/*
 mpic++ -o ./run_mpi ./mpi_helloworld.c
 mpirun -np 4 ./run_mpi
 mpirun -np 4 xterm -e gdb ./run_mpi
 ompi_info
 
 mpicc -o ./run_mpi ./mpi_helloworld.c
 
 export MPIROOT=/usr/local/openmpi
 export PATH=$MPIROOT/bin:$PATH
 export LD_LIBRARY_PATH=$MPIROOT/lib:$LD_LIBRARY_PATH
 export MANPATH=$MPIROOT/share/man:$MANPATH

 */


int main(int argc, char** argv) 
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
}

