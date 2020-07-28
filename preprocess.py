#    ___ _  _ ___ __  __ _____   __
#   / __| || |_ _|  \/  | _ \ \ / /
#  | (__| __ || || |\/| |  _/\ V / 
#   \___|_||_|___|_|  |_|_|   |_| 
# 
# Copyright (c) 2020 Mihaly Kollo. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.

import multiprocessing
from functools import partial
import cupy
import cusignal
from scipy.signal import butter
import numpy as np
from tqdm import trange, tqdm
import re
import importlib
from . import ramdisk
importlib.reload(ramdisk)
import h5py
from chimpy.slurm import Slurm

def cuda_median(a, axis=1):
    a = cupy.asanyarray(a)
    sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]
    if cupy.issubdtype(a.dtype, cupy.inexact):
        kth.append(-1)
    part = cupy.partition(a, kth, axis=axis)
    if part.shape == ():
        return part.item()
    if axis is None:
        axis = 0
    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        indexer[axis] = slice(index, index+1)
    else:
        indexer[axis] = slice(index-1, index+1)
    return cupy.mean(part[indexer], axis=axis)

def filter_traces(s, low_cutoff, high_cutoff, order=3, cmr=False, sample_chunk_size=65536):
    sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')
    n_chunks=s.shape[1]/sample_chunk_size
    chunks=np.hstack((np.arange(n_chunks, dtype=int)*sample_chunk_size,s.shape[1]))
    output=np.empty(s.shape)
    overlap=sample_chunk_size
    chunk=np.zeros((s.shape[0],sample_chunk_size+overlap))
    chunk[:,:overlap]=np.array([s[:,0],]*overlap).transpose()
    for i in trange(len(chunks)-1, ncols=100, position=0, leave=True):
        idx_from=chunks[i]
        idx_to=chunks[i+1]
        chunk=chunk[:,:(idx_to-idx_from+overlap)]
        chunk[:,overlap:]=s[:,idx_from:idx_to]
        cusig=cupy.asarray(chunk, dtype=cupy.float32)
        cusig=cusig-cupy.mean(cusig)
        if cmr:
            cusig=cusig-cuda_median(cusig,0) 
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.flipud(cusig)
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.flipud(cusig)
        output[:,idx_from:idx_to]=cupy.asnumpy(cusig[:,overlap:])
        chunk[:,:overlap]=chunk[:,-overlap:]
    return output

def filter_traces_slurm(recording, stim_recording, low_cutoff, high_cutoff, order=3, cmr=False, sample_chunk_size=65536, samples=-1):
    channels=stim_recording.data['channel']
    if samples==-1:
        fid=h5py.File(recording.file_path, "r")
        n_samples=fid['sig'].shape[1]
        fid.close()
    params={
        'sample_sample_chunk_size':sample_chunk_size,
        'channels':list(map(int, channels)),
        'n_samples':n_samples,
        'cmr':cmr,
        'order':order,
        'low_cutoff':low_cutoff,
        'high_cutoff':high_cutoff,
        'amp_list':list(map(int,1000/stim_recording.data['amp'])),
        'in_file_name': recording.file_path,
        'out_file_name': recording.filtered_filepath
    }
    slurm = Slurm("slurm_filter", 12, gpu=True)
    slurm.run(params)
    

def filter_traces_hdf(recording, stim_recording, low_cutoff, high_cutoff, order=3, cmr=False, sample_chunk_size=65536):
    channels=stim_recording.data['channel']
    channel_scales=cupy.asarray(1000/stim_recording.data['amp'], dtype=cupy.float32)[:,None]
    filtfid=ramdisk.create_hdf_ramfile(recording.filtered_filepath)
    recording.fid.copy('/mapping', filtfid)
    recording.fid.copy('/message_0', filtfid)
    recording.fid.copy('/proc0', filtfid)
    recording.fid.copy('/settings', filtfid)
    recording.fid.copy('/time', filtfid)
    recording.fid.copy('/version', filtfid)
    if 'bits' in recording.fid.keys():
        recording.fid.copy('/bits', filtfid)
    filtfid.create_dataset("sig", (channels.shape[0], recording.fid["sig"].shape[1]), dtype='int16')
    sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')
    n_chunks=recording.fid['sig'].shape[1]/sample_chunk_size
    chunks=np.hstack((np.arange(n_chunks, dtype=int)*sample_chunk_size,recording.fid['sig'].shape[1]))
    overlap=sample_chunk_size
    chunk=np.zeros((channels.shape[0],chunk_sample_chunk_sizesize+overlap))
    chunk[:,:overlap]=np.array([recording.fid['sig'][channels,0],]*overlap).transpose()
    for i in trange(len(chunks)-1, ncols=100, position=0, leave=True):
        idx_from=chunks[i]
        idx_to=chunks[i+1]
        chunk=chunk[:,:(idx_to-idx_from+overlap)]
        chunk[:,overlap:]=recording.fid['sig'][channels,idx_from:idx_to]
        cusig=cupy.asarray(chunk, dtype=cupy.float32)
        cusig=cusig-cupy.mean(cusig)
        if cmr:
            cusig=cusig-cuda_median(cusig,0) 
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.flipud(cusig)
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.flipud(cusig)
        cusig=cusig*channel_scales
        filtfid["sig"][:,idx_from:idx_to]=cupy.asnumpy(cusig[:,overlap:])
        chunk[:,:overlap]=chunk[:,-overlap:]
    print("Writing filtered traces to disk...")
    ramdisk.save_ramfile(recording.filtered_filepath)
    return filtfid

def get_spike_amps(s):
    mean_stim_trace=cupy.asnumpy(cupy.mean(s,axis=0));
    spike_threshold=-cupy.std(mean_stim_trace)*4;
    crossings=np.where(mean_stim_trace<spike_threshold)[0][:-2];
    amps=np.abs(cupy.asnumpy(cupy.mean(s[:,crossings[np.diff(crossings, prepend=0)>1]+2],axis=1)))
    return np.vstack((np.where(amps>50),amps[amps>50]))    