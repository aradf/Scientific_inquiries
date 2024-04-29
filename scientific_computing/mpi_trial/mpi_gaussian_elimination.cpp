// MPI parallel gaussian elimination

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

#include "mpi.h"

/*
 mpic++ -o ./run_mpi mpi_gaussian_elimination.cpp
 mpirun -np 4 ./run_mpi
 mpirun -np 4 xterm -e gdb ./run_mpi
 ompi_info

 export MPIROOT=/usr/local/openmpi
 export PATH=$MPIROOT/bin:$PATH
 export LD_LIBRARY_PATH=$MPIROOT/lib:$LD_LIBRARY_PATH
 export MANPATH=$MPIROOT/share/man:$MANPATH

 */

void print_matrix(const float *matrix, int dim) 
{
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      std::cout << matrix[i * dim + j] << ' ';
    }
    std::cout << '\n';
  }
}

int main(int argc, char *argv[]) {
  // Initialize MPI
  MPI_Init(&argc, 
           &argv);

  // Get the total number of tasks
  int number_tasks;
  MPI_Comm_size(MPI_COMM_WORLD, 
                &number_tasks);

  // Calculate the number of rows mapped to each process
  // Assumes this divides evenly.  2^4 = 16
  const int dim = 1 << 4;
  const int number_rows = dim / number_tasks;

  // Get the task ID
  int task_id;
  MPI_Comm_rank(MPI_COMM_WORLD, 
                &task_id);

  const int start_row = task_id * number_rows;
  const int end_row = start_row + number_rows;

  // Matrix - Only initialized in rank 0
  // matrix is a unique pointer object owning an array of float.
  std::unique_ptr<float[]> matrix;

  // Each process will store a chunk of the matrix
  // Allocate memory for the array
  std::unique_ptr<float []> m_chunk;
  m_chunk = std::make_unique<float[]>(dim * number_rows);

  // Each process will receive a pivot row each iteration
  std::unique_ptr<float[]> pivot_row;
  pivot_row = std::make_unique<float[]>(dim);

  // Only rank 0 create/initializes the matrix
  if (task_id == 0) 
  {
    // Create a random number generator
    std::mt19937 mt(123);
    std::uniform_real_distribution<> dist(1.0f, 
                                          2.0f);

    // Create a matrix
    matrix = std::make_unique<float[]>(dim * dim);
    std::generate(matrix.get(), 
                  matrix.get() + dim * dim,
                  [&] { return dist(mt); });

   print_matrix(matrix.get(), 
                dim);

  }

  std::cout << "P:" << task_id << std::endl;

  // Before doing anything, send parts of the matrix to each process
  MPI_Scatter(matrix.get(), 
              dim * number_rows, 
              MPI_FLOAT, 
              m_chunk.get(),
              dim * number_rows, 
              MPI_FLOAT, 
              0, 
              MPI_COMM_WORLD);

  // Store requests that for non-blocking sends
  std::vector<MPI_Request> requests(number_tasks);

  // Performance gaussian elimination
  for (int row = 0; 
       row < end_row; 
       row++) 
  {
    // See if this process is responsible for the pivot calculation
    auto mapped_rank = row / number_rows;

    // If the row is mapped to this rank...
    if (task_id == mapped_rank) 
    {
      // Calculate the row in the local matrix
      auto local_row = row % number_rows;

      // Get the value of the pivot
      auto pivot = m_chunk[local_row * dim + row];

      // Divide the rest of the row by the pivot
      for (int col = row; 
           col < dim; 
           col++) 
      {
        m_chunk[local_row * dim + col] /= pivot;
      }

      // Send the pivot row to the other processes
      for (int i = mapped_rank + 1; 
           i < number_tasks; 
           i++) 
      {
        MPI_Isend(m_chunk.get() + dim * local_row, 
                  dim, 
                  MPI_FLOAT, 
                  i, 
                  0,
                  MPI_COMM_WORLD, 
                  &requests[i]);
      }

      // Eliminate the for the local rows
      for (int eliminate_row = local_row + 1; 
               eliminate_row < number_rows; 
               eliminate_row++) 
      {
        // Get the scaling factor for elimination
        auto scale = m_chunk[eliminate_row * dim + row];

        // Remove the pivot
        for (int col = row; col < dim; col++) 
        {
          m_chunk[eliminate_row * dim + col] -= m_chunk[local_row * dim + col] * scale;
        }
      }

      // Check if there are any outstanding messages
      for (int i = mapped_rank + 1; 
               i < number_tasks; 
               i++) 
      {
        MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
      }
    } else 
      {
      // Receive pivot row
      MPI_Recv(pivot_row.get(), 
               dim, 
               MPI_FLOAT, 
               mapped_rank, 
               0, 
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      // Skip rows that have been fully processed
      for (int eliminate_row = 0; 
               eliminate_row < number_rows; 
               eliminate_row++) 
      {
        // Get the scaling factor for elimination
        auto scale = m_chunk[eliminate_row * dim + row];

        // Remove the pivot
        for (int col = row; 
                 col < dim; 
                 col++) 
        {
          m_chunk[eliminate_row * dim + col] -= pivot_row[col] * scale;
        }
      }
    }
  }

  // Gather the final results into rank 0
  MPI_Gather(m_chunk.get(), 
             number_rows * dim, 
             MPI_FLOAT, 
             matrix.get(), 
             number_rows * dim,
             MPI_FLOAT, 
             0, 
             MPI_COMM_WORLD);

  if (task_id == 0)
  {
     std::cout << "P:" << task_id << "Matrix: " << std::endl;
       print_matrix(matrix.get(), 
                dim);

  }

  // Finish our MPI work
  MPI_Finalize();
  return 0;
}
