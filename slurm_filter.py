#    ___ _  _ ___ __  __ _____   __
#   / __| || |_ _|  \/  | _ \ \ / /
#  | (__| __ || || |\/| |  _/\ V / 
#   \___|_||_|___|_|  |_|_|   |_| 
# 
# Copyright (c) 2020 Mihaly Kollo. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.                                 

import numpy as np
import sys
import h5py
from mpi4py import MPI
import time
import os
import cupy
import cusignal

MASTER_PROCESS = 0
WORK_TAG = 1
DIE_TAG = 2
sample_batch_size=65536

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()
inode = MPI.Get_processor_name()
print(nproc)
print(iproc)

# if iproc == MASTER_PROCESS:
#     in_file = h5py.File(sys.argv[2], 'r', driver='mpio', comm=MPI.COMM_WORLD)
#     out_file = h5py.File(sys.argv[2].split('.raw.h5')[0]+".filt.h5", 'w', driver='mpio', comm=MPI.COMM_WORLD)
#     out_dset = out_file.create_dataset('sig', in_file['sig'].shape, dtype='int16')
#     out_dset[[1024, 1025, 1026, 1027],:]=in_file['sig'][[1024, 1025, 1026, 1027],:]
#     channels=np.arange(1024)
#     channel_chunks=np.array_split(channels, nproc-1)
#     n_samples=in_file['sig'].shape[1]
#     n_sample_chunks=n_samples/batch_size
#     sample_chunk_borders=np.hstack((np.arange(n_sample_chunks, dtype=int)*batch_size,n_samples))
#     sample_chunks=dict(zip(np.arange(nproc-1),[np.array([sample_chunk_borders[i:i+2].copy() for i in range(len(sample_chunk_borders)-1)])]*(nproc-1)))
#     current_chunks=dict(zip(np.arange(11),[None]*(nproc-1)))
#     n_all_chunks=sum([sample_chunks[i].shape[0] for i in np.arange(nproc-1)])
#     n_total_chunks=sum([sample_chunks[i].shape[0] for i in np.arange(3)])
#     while (sum([sample_chunks[i].shape[0] for i in np.arange(nproc-1)])+sum([cc is not None for cc in current_chunks]))>0:
#         print(sum([sample_chunks[i].shape[0] for i in np.arange(3)])*1000/n_total_chunks)
#         for i, ch in enumerate(current_chunks):
#             if ch==None and sample_chunks[i].shape[0]>0:
#                 batch=np.zeros((channel_chunks[i].shape[0],batch_size*2))
#                 current_chunks[i]=sample_chunks[i][0]
#                 sample_chunks[i]=np.delete(sample_chunks[i], (0), axis=0)
#                 if sample_chunks[i][0][0]==0:
#                     batch[:,:batch_size]=np.array([in_file['sig'][channel_chunks[i],0],]*batch_size).transpose()
#                 else:
#                     batch[:,:batch_size]=in_file['sig'][channel_chunks[i],(current_chunks[i][0]-batch_size):(current_chunks[i][1]-batch_size)]
#                 batch[:,batch_size:]=in_file['sig'][channel_chunks[i],current_chunks[i][0]:current_chunks[i][1]]
#                 comm.Send([data, MPI.SHORT], dest=i, tag=WORK_TAG)
#         data = numpy.empty(batch_size, dtype='int16')
#         status = MPI.Status()
#         comm.Recv([data, MPI.SHORT], source=comm.ANY_SOURCE, tag=comm.ANY_TAG, status=status)
#         rnk = status.Get_source()
#         out_dset[channel_chunks[i],current_chunks[rnk][0]:current_chunks[rnk][1]]=data
#         current_chunks[rnk]=None
#     for proc in range(nproc-1):
#         comm.Send(-1, proc, tag=DIE_TAG)     
#     print(int(1000), flush=True)
#     print('DONE')
#     in_file.close()
#     out_file.close() 
# else:
#     continue_working = True
#     while continue_working:
#         status = MPI.Status()
#         comm.Recv(source=MASTER_PROCESS, tag=comm.ANY_TAG, status=status)
#         if status.tag == DIE_TAG:
#             continue_working = False
#         else:
#             batch = numpy.empty(batch_size*2, dtype='int16')
#             work_array, status = comm.Receive(source=MASTER_PROCESS, tag=pypar.ANY_TAG,
#                 return_status=True)       
#             cusig=cupy.asarray(batch, dtype=cupy.float32)
#             cusig=cusig-cupy.mean(cusig)
#             if cmr:
#                 cusig=cusig-cuda_median(cusig,0) 
#             cusig=cusignal.sosfilt(sos,cusig)
#             cusig=cupy.flipud(cusig)
#             cusig=cusignal.sosfilt(sos,cusig)
#             cusig=cupy.flipud(cusig)
#             cusig=cusig*channel_scales
#             result_array=cupy.asnumpy(cusig[:,overlap:])
#             comm.Send(result_array, destination=MASTER_PROCESS, tag=iproc)
   
MPI.Finalize()

