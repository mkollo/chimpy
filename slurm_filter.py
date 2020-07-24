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
SETUP_TAG = 1
WORK_TAG = 2
DIE_TAG = 3

sample_batch_size=65536
cmr=False
order=3
low_cutoff=100
high_cutoff=9000
amp_list=np.ones((1024))*250
channel_scales=1000/amp_list

sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()
inode = MPI.Get_processor_name()
status = MPI.Status()

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
    n_all_chunks=sum([sample_chunks[i].shape[0] for i in np.arange(nproc-1)])
    n_total_chunks=sum([sample_chunks[i].shape[0] for i in np.arange(3)])
    for i, channel_per_proc in enumerate([len(c) for c in channel_chunks]):
        comm.send(channel_per_proc, dest=i+1, tag=SETUP_TAG)
    while (sum([sample_chunks[i].shape[0] for i in np.arange(nproc-1)])+sum([cc is not None for cc in current_chunks]))>0:
        print((1-sum([sample_chunks[i].shape[0] for i in np.arange(3)])/n_total_chunks)*1000, flush=True)
        print(current_chunks, flush=True)
        for i, _ in enumerate(current_chunks):
            ch=current_chunks[i]
            print("current chunk for "+str(i)+" is "+str(ch), flush=True)
            print("all chunks for "+str(i)+" are "+ str(sample_chunks[i].shape[0]), flush=True)
            print("Sample chunks reamining size: "+str(sample_chunks[i]), flush=True)
            if ch==None and sample_chunks[i].shape[0]>0:
                batch=np.zeros((channel_chunks[i].shape[0],sample_batch_size*2+1))
                current_chunks[i]=sample_chunks[i][0]
                sample_chunks[i]=np.delete(sample_chunks[i], (0), axis=0)
                print("Master made chunks", flush=True)
                if current_chunks[i][0]==0:
                    batch[:,:sample_batch_size]=np.array([in_file['sig'][channel_chunks[i],0],]*sample_batch_size).transpose()
                else:
                    batch[:,:sample_batch_size]=in_file['sig'][channel_chunks[i],]
                batch[:,sample_batch_size:-1]=in_file['sig'][channel_chunks[i],current_chunks[i][0]:current_chunks[i][1]]
                batch[:,-1]=channel_scales[channel_chunks[i]]
                print("Master loaded batch from file")
                print("Master sending data to proc "+str(i+1) + " of size "+str(batch.shape), flush=True)
                comm.send(batch, dest=i+1, tag=WORK_TAG)
                print("Master sent data to proc "+str(i+1), flush=True)
        data=comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        rnk = status.Get_source()
        print("Dset shape "+str(out_dset[channel_chunks[i],current_chunks[rnk-1][0]:current_chunks[rnk-1][1]].shape),flush=True)
        print("sample_batch_size "+str(sample_batch_size),flush=True)
        print("Data shape 3 "+str(data.shape),flush=True)
        out_dset[channel_chunks[rnk-1],current_chunks[rnk-1][0]:current_chunks[rnk-1][1]]=data
        current_chunks[rnk-1]=None
    for proc in range(nproc-1):
        comm.send(-1, proc, tag=DIE_TAG)    
else: 
    print("Process "+str(iproc)+" waiting for setup", flush=True)
    proc_n_channels=comm.recv(source=0, tag=SETUP_TAG)
    print("Process "+str(iproc)+" received setup: "+str(proc_n_channels))
    batch = np.empty((proc_n_channels,sample_batch_size*2+1), dtype='int16')
    continue_working = True
    while continue_working:
        print("Process "+str(iproc)+" waiting for data of size "+str(batch.shape), flush=True)
        batch=comm.recv(source=0, tag=WORK_TAG, status=status)    
        tag = status.Get_tag()
        print("Process "+str(iproc)+" received data", flush=True)
        if tag == DIE_TAG:
            continue_working = False
        else:
            print("Proc "+str(iproc)+" received data", flush=True)
            cusig=cupy.asarray(batch[:,:-1], dtype=cupy.float32)
            cusig=cusig-cupy.mean(cusig)
            if cmr:
                cusig=cusig-cuda_median(cusig,0) 
            cusig=cusignal.sosfilt(sos,cusig)
            cusig=cupy.flipud(cusig)
            cusig=cusignal.sosfilt(sos,cusig)
            cusig=cupy.flipud(cusig)
            proc_channel_scales=cupy.asarray(batch[:,-1], dtype=cupy.float32)[:,None]
            cusig=cusig*proc_channel_scales
            result_array=cupy.asnumpy(cusig[:,int((batch.shape[1]-1)/2):])
            print("Proc "+str(iproc)+" calculated result", flush=True)
            print("Batch shape " + str((batch.shape[1]-1)/2),flush=True)

            print("Result array size "+str(result_array.shape),flush=True)
            comm.send(result_array, dest=MASTER_PROCESS, tag=iproc)
            print("Proc "+str(iproc)+" sent result", flush=True)

            
comm.Barrier()

if iproc == MASTER_PROCESS:
    print(int(1000), flush=True)
    print('DONE', flush=True)
    in_file.close()
    out_file.close() 
    
MPI.Finalize()

