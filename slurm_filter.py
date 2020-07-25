#    ___ _  _ ___ __  __ _____   __
#   / __| || |_ _|  \/  | _ \ \ / /
#  | (__| __ || || |\/| |  _/\ V / 
#   \___|_||_|___|_|  |_|_|   |_| 
# 
# Copyright (c) 2020 Mihaly Kollo. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.                                 

import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import numpy as np
import sys
import h5py
import time
import os

from scipy.signal import butter
import cupy
import cusignal

MASTER_PROCESS = 0
SHAPE_TAG = 71
AMPS_TAG = 72
DATA_TAG = 73
DIE_TAG = 74

sample_batch_size=65536
cmr=False
order=3
low_cutoff=100
high_cutoff=9000
amp_list=np.ones((1024))*250
channel_scales=1000/amp_list


comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()
inode = MPI.Get_processor_name()
status = MPI.Status()

sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')

comm.Barrier()
if iproc == MASTER_PROCESS:
    in_file = h5py.File(sys.argv[1], 'r')
    out_file = h5py.File(sys.argv[1].split('.raw.h5')[0]+".filt.h5", 'w')
    out_dset = out_file.create_dataset('sig', in_file['sig'].shape, dtype='float32')
    out_dset[[1024, 1025, 1026, 1027],:]=in_file['sig'][[1024, 1025, 1026, 1027],:]
    channels=np.arange(1024)
    channel_chunks=np.array_split(channels, nproc-1)
    n_samples=in_file['sig'].shape[1]
    n_sample_chunks=n_samples/sample_batch_size
    sample_chunk_borders=np.hstack((np.arange(n_sample_chunks, dtype=int)*sample_batch_size,n_samples))
    sample_chunks=dict(zip(np.arange(nproc-1),[np.array([sample_chunk_borders[i:i+2].copy() for i in range(len(sample_chunk_borders)-1)])]*(nproc-1)))
    current_chunks=dict(zip(np.arange(11),[None]*(nproc-1)))
    n_total_chunks=sum([sample_chunks[i].shape[0] for i in np.arange(nproc-1)])
    n_remaining_chunks=n_total_chunks
    n_current_chunks=0
    while (n_remaining_chunks+n_current_chunks)>0:
        for i, _ in enumerate(current_chunks):
            if current_chunks[i] is None and sample_chunks[i].shape[0]>0:
                current_chunks[i]=sample_chunks[i][0]
                data_shape=(channel_chunks[i].shape[0], sample_batch_size+current_chunks[i][1]-current_chunks[i][0])
                comm.send(data_shape, dest=i+1, tag=SHAPE_TAG)
                comm.send(channel_scales[channel_chunks[i]], dest=i+1, tag=AMPS_TAG)
                batch=np.empty((channel_chunks[i].shape[0],sample_batch_size+current_chunks[i][1]-current_chunks[i][0]))
                if current_chunks[i][0]==0:
                    batch[:,:sample_batch_size]=np.array([in_file['sig'][channel_chunks[i],0],]*sample_batch_size).transpose()
                else:
                    batch[:,:sample_batch_size]=in_file['sig'][channel_chunks[i],(current_chunks[i][0]-sample_batch_size):current_chunks[i][0]]
                sample_chunks[i]=np.delete(sample_chunks[i], (0), axis=0)
                batch[:,sample_batch_size:]=in_file['sig'][channel_chunks[i],current_chunks[i][0]:current_chunks[i][1]]
                comm.send(batch, dest=i+1, tag=DATA_TAG)
                
        data=comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        rnk = status.Get_source()
        print("Data from "+str(rnk)+" shape: "+str(data.shape))
        print("Data from "+str(rnk)+" out  : "+str(out_dset[channel_chunks[rnk-1],current_chunks[rnk-1][0]:current_chunks[rnk-1][1]].shape))

        out_dset[channel_chunks[rnk-1],current_chunks[rnk-1][0]:current_chunks[rnk-1][1]]=data
        current_chunks[rnk-1]=None
        print(sum([sample_chunks[i].shape[0] for i in np.arange(nproc-1)])+sum([cc is not None for cc in current_chunks]))
        print(sum([sample_chunks[i].shape[0] for i in np.arange(nproc-1)]))
        print(sum([cc is not None for cc in current_chunks]))
        print(current_chunks)
        n_current_chunks=sum([current_chunks[cc] is not None for cc in current_chunks])
        n_remaining_chunks=sum([sample_chunks[i].shape[0] for i in np.arange(nproc-1)])
        
    for proc in range(nproc-1):
        comm.send(-1, proc, tag=DIE_TAG)    
else: 
    continue_working = True
    while continue_working:
        batch_shape=comm.recv(source=MASTER_PROCESS, tag=SHAPE_TAG)    
        amps=comm.recv(source=MASTER_PROCESS, tag=AMPS_TAG)            
        batch=comm.recv(source=0, tag=DATA_TAG, status=status)    
        tag = status.Get_tag()
        if tag == DIE_TAG:
            continue_working = False
        else:
            cusig=cupy.asarray(batch, dtype=cupy.float32)
            cusig=cusig-cupy.mean(cusig)
            if cmr:
                cusig=cusig-cuda_median(cusig,0) 
            cusig=cusignal.sosfilt(sos,cusig)
            cusig=cupy.flipud(cusig)
            cusig=cusignal.sosfilt(sos,cusig)
            cusig=cupy.flipud(cusig)
            proc_channel_scales=cupy.asarray(batch[:,-1], dtype=cupy.float32)[:,None]
            cusig=cusig*proc_channel_scales
            result_array=cupy.asnumpy(cusig[:,-(batch.shape[1]-sample_batch_size):])
            comm.send(result_array, dest=MASTER_PROCESS, tag=iproc)

if iproc == MASTER_PROCESS:
    print(int(1000), flush=True)
    print('DONE', flush=True)
    in_file.close()
    out_file.close() 
    
MPI.Finalize()

