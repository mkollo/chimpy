#!/usr/bin/env python
import numpy as np
import sys
import h5py
from mpi4py import MPI
import time
import os

nproc = MPI.COMM_WORLD.Get_size()   # Size of communicator
iproc = MPI.COMM_WORLD.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs

in_file = h5py.File('/camp/home/kollom/working/mkollo/CHIME/BR_200713/stim/BR_200713_stim_abovebrain.raw.h5', 'r')
out_file = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
out_file.atomic = True
out_dset = out_file.create_dataset('sig', (1024,1000), dtype='int16')

channel=int(sys.argv[1])

for i in range(0,nproc):
    MPI.COMM_WORLD.Barrier()
    if iproc == i:
        values = in_file['sig'][channel,:1000]
        iterations=100
        for j in range(iterations):
            out_dset[channel,:]=values-np.mean(values)+1
            print(int(j/iterations*1000), flush=True)
        print('DONE')
in_file.close()
out_file.close()
MPI.Finalize()

