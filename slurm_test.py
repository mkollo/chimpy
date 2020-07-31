import mpi4py
mpi4py.rc.recv_mprobe = False
mpi4py.rc.threads = False
from mpi4py import MPI

import json
import numpy as np
import subprocess

MASTER_PROCESS=0

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()
inode = MPI.Get_processor_name()
status = MPI.Status()

# comm.Barrier()
if iproc == MASTER_PROCESS:
    print("I'm a master process")
    comm.send("Sent message", dest=1, tag=0)
    print(comm.recv(source=1, tag=0))
else: 
    print("I'm a slave process")
    print(comm.recv(source=0, tag=0))
    comm.send("Sent message back", dest=0, tag=0)
