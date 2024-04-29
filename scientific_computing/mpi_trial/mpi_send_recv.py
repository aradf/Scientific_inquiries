#!/usr/bin/env python
import numpy
from mpi4py import MPI

'''
mpirun -np 4 python3 mpi_send_recv.py
sudo apt install python3-venv
source mpi-env/bin/activate
python3 -m pip install mpi4py
python3 -m pip install numpy
'''

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
test_numpyArray = numpy.array([rank]*5, 
                              dtype=float)

print(f'P:{rank} hellow world: ')

if comm.rank==0:
    comm.send(test_numpyArray,
              dest=(rank+1%size))

if comm.rank > 0:
    data=comm.recv(source=(rank-1)%size)
    comm.send(test_numpyArray,
              dest=(rank+1)%size)

if comm.rank==0:
    data=comm.recv(source=size-1)

print(f'P:{rank} Received this: {data}')

