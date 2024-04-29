// Serial gaussian elimination

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

/*
 g++ -g -o ./run_serial serial_gaussian_elimination.cpp
 cd /usr/include/
 find /usr/include -name iostream -type f -print
 cd /usr/include/c++/9
*/

// Helper function to print the matrix
void print_matrix(const float *matrix, int dim) 
{
  for (int i = 0; i < dim; i++) 
  {
    for (int j = 0; j < dim; j++) 
    {
      std::cout << matrix[i * dim + j] << ' ';
    }
    std::cout << '\n';
  }
}

int main() 
{
  // Create a random number generator
  std::mt19937 mt(123);
  std::uniform_real_distribution<> dist(1.0f, 2.0f);

  // Create a matrix
  std::vector<float> matrix;
  // const int dim = 1 << 12;
  const int dim = 1 << 4;
  std::generate_n(std::back_inserter(matrix), 
                  dim * dim, 
                  [&] { return dist(mt); });

  print_matrix(matrix.data(), dim);

  // Performance gaussian elimination
  for (int row = 0; 
           row < dim; 
           row++) 
  {
    // Get the value of the pivot
    auto pivot = matrix[row * dim + row];

    // Divide the rest of the row by the pivot
    for (int col = row; 
             col < dim; 
             col++) 
    {
      matrix[row * dim + col] /= pivot;
    }

    // Eliminate the pivot col from the remaining rows
    for (int eliminate_row = row + 1; 
             eliminate_row < dim; 
             eliminate_row++) 
    {
      // Get the scaling factor for elimination
      auto scale = matrix[eliminate_row * dim + row];

      // Remove the pivot
      for (int col = row; 
               col < dim; 
               col++) 
      {
        matrix[eliminate_row * dim + col] -= matrix[row * dim + col] * scale;
      }
    }
  }
  print_matrix(matrix.data(), dim);
  return 0;
}