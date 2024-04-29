#!/usr/bin/env python
from mpi4py import MPI

'''
mpirun -np 4 python3 mpi_helloworld.py
sudo apt install python3-venv
source mpi-env/bin/activate
python3 -m pip install mpi4py
'''

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f'P:{rank} Hello World!')
