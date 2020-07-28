import mpi4py
mpi4py.rc.recv_mprobe = False
mpi4py.rc.threads = False
from mpi4py import MPI

import json
import numpy as np
# import h5py
import subprocess

MASTER_PROCESS=0

with open('params.json') as json_file:
    params = json.load(json_file)
    
sample_batch_size=params['sample_batch_size']
channels=np.array(params['channels'])
n_samples=params['n_samples']
cmr=params['cmr']
order=params['order']
low_cutoff=params['low_cutoff']
high_cutoff=params['high_cutoff']
amp_list=np.array(params['amp_list'])
in_file_name=params['in_file_name']
out_file_name=params['out_file_name']

channel_scales=1000/amp_list
n_channels=channels.shape[0]

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()
inode = MPI.Get_processor_name()
status = MPI.Status()

comm.Barrier()
print(subprocess.check_output(['ls','/dev/shm']).decode("utf-8"))
if iproc == MASTER_PROCESS:
#     in_file = h5py.File(in_file_name, 'r')
    print("I'm a master process")
else: 
    print("I'm a slave process")
