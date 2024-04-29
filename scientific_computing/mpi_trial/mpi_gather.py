#!/usr/bin/env python
import numpy
from mpi4py import MPI

'''
mpirun -np 4 python3 mpi_gather.py
sudo apt install python3-venv
source mpi-env/bin/activate
python3 -m pip install mpi4py
python3 -m pip install numpy
'''

send_buffer = []
root = 0

### initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f'P:{comm.rank} Hello world...')

if comm.rank == 0:
    numpy_array=numpy.array(range(comm.size*comm.size), 
                            dtype=float)
    
    numpy_array.shape=(comm.size,
             comm.size)
    
    print(f'P:{comm.rank} - {numpy_array}')
    send_buffer = numpy_array

my_vector = comm.scatter(send_buffer, 
                         root)

print(f'P:{comm.rank} Got this array: {my_vector}')

my_vector = my_vector * my_vector

recv_buffer = comm.gather(my_vector,
                          root)

if comm.rank == 0:
    print(f'P:{comm.rank} - {numpy.array(recv_buffer)}')


