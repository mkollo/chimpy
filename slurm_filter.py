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
mpi4py.rc.threads = False
from mpi4py import MPI

import numpy as np
import sys
import h5py
import time
import os
import json

from scipy.signal import butter
import cupy
import cusignal

MASTER_PROCESS = 0
SHAPE_TAG = 71
AMPS_TAG = 72
DATA_TAG = 73
DIE_TAG = 74


# Load filter parameters

with open('params.json') as json_file:
    params = json.load(json_file)
    
sample_chunk_size=params['sample_batch_size']
channels=np.array(params['channels'])
n_samples=params['n_samples']
cmr=params['cmr']
order=params['order']
low_cutoff=params['low_cutoff']
high_cutoff=params['high_cutoff']
amp_list=np.array(params['amp_list'])
in_file_name=params['in_file_name']
out_file_name=params['out_file_name']

# Create dictionary of channel scales
channel_scales=dict(zip(channels,1000/amp_list))
n_channels=channels.shape[0]

# Initialize MPI
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()
inode = MPI.Get_processor_name()
status = MPI.Status()

# Calculate filter
sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')

comm.Barrier()
if iproc == MASTER_PROCESS:
#     Load input file
    in_file = h5py.File(in_file_name, 'r')
    if n_samples<0:
        n_samples=in_file['sig'].shape[1]
#     Create output file
    out_file = h5py.File(out_file_name, 'w')
    out_dset = out_file.create_dataset('sig', (1024, n_samples), dtype='float32')
    
    out_file.create_dataset('/mapping', data=in_file['mapping'])  
    
    # Renumber channels for storing data in n_samples x channels 'sig' matrix instead of the original n_samples x 1024 matrix
    for ic, _ in enumerate(channels):
        element=in_file['mapping'][channels[ic]]
        element[0]=ic
        out_file['mapping'][ic]=element    
#     Create data chunks (channels and samples) for each MPI process
    in_channel_chunks=np.array_split(channels, nproc-1)
    out_channel_chunks=np.array_split(list(range(channels)), nproc-1)
    n_sample_chunks=n_samples/sample_chunk_size
    sample_chunk_borders=np.hstack((np.arange(n_sample_chunks, dtype=int)*sample_chunk_size,n_samples))
    sample_chunks=dict(zip(np.arange(nproc-1),[np.array([sample_chunk_borders[i:i+2].copy() for i in range(len(sample_chunk_borders)-1)])]*(nproc-1)))
    # Dictionary for holding currently processed chunks for each process
    current_chunks=dict(zip(np.arange(11),[None]*(nproc-1)))
    n_total_chunks=sum([sample_chunks[i].shape[0] for i in np.arange(nproc-1)])
    n_remaining_chunks=n_total_chunks
    n_current_chunks=0
#     Master process main loop
    while (n_remaining_chunks+n_current_chunks)>0:
#         Checking for idle processes, and delegating chunks
        for i, _ in enumerate(current_chunks):
            if current_chunks[i] is None and sample_chunks[i].shape[0]>0:
                current_chunks[i]=sample_chunks[i][0]
                data_shape=(in_channel_chunks[i].shape[0], sample_chunk_size+current_chunks[i][1]-current_chunks[i][0])
                comm.send(data_shape, dest=i+1, tag=SHAPE_TAG)
                comm.send([channel_scales[channel] for channel in in_channel_chunks[i]], dest=i+1, tag=AMPS_TAG)
                chunk=np.empty((in_channel_chunks[i].shape[0],sample_chunk_size+current_chunks[i][1]-current_chunks[i][0]))
                if current_chunks[i][0]==0:
                    chunk[:,:sample_chunk_size]=np.array([in_file['sig'][in_channel_chunks[i],0],]*sample_chunk_size).transpose()
                else:
                    chunk[:,:sample_chunk_size]=in_file['sig'][in_channel_chunks[i],(current_chunks[i][0]-sample_chunk_size):current_chunks[i][0]]
                sample_chunks[i]=np.delete(sample_chunks[i], (0), axis=0)
                chunk[:,sample_chunk_size:]=in_file['sig'][in_channel_chunks[i],current_chunks[i][0]:current_chunks[i][1]]
                comm.send(chunk, dest=i+1, tag=DATA_TAG)
#         Waiting for next ready chunk
        data=comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        rnk = status.Get_source()
#         Writing results to output file
        out_dset[out_channel_chunks[rnk-1],current_chunks[rnk-1][0]:current_chunks[rnk-1][1]]=data
        current_chunks[rnk-1]=None
        n_current_chunks=sum([current_chunks[cc] is not None for cc in current_chunks])
        n_remaining_chunks=sum([sample_chunks[i].shape[0] for i in np.arange(nproc-1)])
#         Reportint progress
        print((1-n_remaining_chunks/n_total_chunks)*999, flush=True)
#     After finishing the main loop, killing processes
    for proc in range(nproc-1):
        comm.send(-1, proc, tag=DIE_TAG)    
else: 
#     Slave process main loop
    continue_working = True
    while continue_working:
#         Waiting for data from master process
        chunk_shape=comm.recv(source=MASTER_PROCESS, tag=SHAPE_TAG)    
        amps=comm.recv(source=MASTER_PROCESS, tag=AMPS_TAG)            
        chunk=comm.recv(source=0, tag=DATA_TAG, status=status)    
        tag = status.Get_tag()
        if tag == DIE_TAG:
            continue_working = False
        else:
            cusig=cupy.asarray(chunk, dtype=cupy.float32)
            cusig=cusig-cupy.mean(cusig)
            if cmr:
                cusig=cusig-cuda_median(cusig,0) 
            cusig=cusignal.sosfilt(sos,cusig)
            cusig=cupy.flipud(cusig)
            cusig=cusignal.sosfilt(sos,cusig)
            cusig=cupy.flipud(cusig)
            proc_channel_scales=cupy.asarray(chunk[:,-1], dtype=cupy.float32)[:,None]
            cusig=cusig*proc_channel_scales
            result_array=cupy.asnumpy(cusig[:,-(chunk.shape[1]-sample_chunk_size):])
            comm.send(result_array, dest=MASTER_PROCESS, tag=iproc)

if iproc == MASTER_PROCESS:
    in_file.close()
    out_file.close() 
    print('DONE', flush=True)

