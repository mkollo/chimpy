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

def filter_traces(s, low_cutoff, high_cutoff, order=3, cmr=False, sample_chunk_size=65536, n_samples=-1):
    if n_samples==-1:
        n_samples=s.shape[1]
    sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')
    n_chunks=n_samples/sample_chunk_size
    chunks=np.hstack((np.arange(n_chunks, dtype=int)*sample_chunk_size,n_samples))
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

def filter_experiment_slurm(experiment, low_cutoff, high_cutoff, order=3, cmr=False, sample_chunk_size=65536, n_samples=-1):
    channels=stim_recording.data['channel']
    if n_samples==-1:
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
    

def filter_experiment_local(experiment, low_cutoff, high_cutoff, order=3, cmr=False, sample_chunk_size=65536, n_samples=-1, ram_copy=False):
    stim_recording=experiment.recordings['stim']
    brain_recording=experiment.recordings['brain']
    channels=np.array(stim_recording.channels)[experiment.strong_pixels]
    amps=np.array(stim_recording.amps)[stim_recording.strong_pixels]
    scales=1000/amps
#     Optionally save file into a tmpfs partition for processing
    if ram_copy:
        in_ramfile=RamFile(brain_recording.filepath, 'r')
        in_filepath=in_ramfile.file_path
        out_ramfile=RamFile(brain_recording.filtered_filepath, 'w')
        out_filepath=out_ramfile.file_path
    else:
        in_filepath=brain_recording.filepath
        out_filepath=brain_recording.filtered_filepath
#     Create output file
    in_fid=h5py.File(in_filepath, 'r')
    out_fid=h5py.File(out_filepath, 'w')
    if n_samples==-1:
        n_samples=in_fid['sig'].shape[1]
    in_fid.copy('/mapping', out_fid)
    in_fid.copy('/message_0', out_fid)
    in_fid.copy('/proc0', out_fid)
    in_fid.copy('/settings', out_fid)
    in_fid.copy('/time', out_fid)
    in_fid.copy('/version', out_fid)
    if 'bits' in in_fid.keys():
        in_fid.copy('/bits', out_fid)
    out_fid.create_dataset("sig", (channels.shape[0], n_samples), dtype='int16')
#     Create filter: cutoff / 0.5 * fs
    sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')
#     Create chunks
    n_chunks=n_samples/sample_chunk_size
    sample_chunks=np.hstack((np.arange(n_chunks, dtype=int)*sample_chunk_size,n_samples))
    overlap=sample_chunk_size
    chunk=np.zeros((channels.shape[0],sample_chunk_size+overlap))
    chunk[:,:overlap]=np.array([list(range(channels.shape[0])),]*overlap).transpose()
    for i in trange(len(sample_chunks)-1, ncols=100, position=0, leave=True):
        idx_from=sample_chunks[i]
        idx_to=sample_chunks[i+1]
        chunk=chunk[:,:(idx_to-idx_from+overlap)]
        chunk[:,overlap:]=in_fid['sig'][channels,idx_from:idx_to]
        cusig=cupy.asarray(chunk, dtype=cupy.float32)
        cusig=cusig-cupy.mean(cusig)
        if cmr:
            cusig=cusig-cuda_median(cusig,0) 
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.flipud(cusig)
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.flipud(cusig)
        cusig=cusig*cupy.asarray(scales, dtype=cupy.float32)[:,None]
        out_fid["sig"][:,idx_from:idx_to]=cupy.asnumpy(cusig[:,overlap:])
        chunk[:,:overlap]=chunk[:,-overlap:]
#     Writing filtered traces to disk...
    in_fid.close()
    out_fid.close()
    if ram_copy:
        in_ramfile.save()
        out_ramfile.save()
        del in_ramfile, out_ramfile
    return out_filepath

def get_spike_amps(s):
    mean_stim_trace=cupy.asnumpy(cupy.mean(s,axis=0));
    spike_threshold=-cupy.std(mean_stim_trace)*4;
    crossings=np.where(mean_stim_trace<spike_threshold)[0][:-2];
    amps=np.abs(cupy.asnumpy(cupy.mean(s[:,crossings[np.diff(crossings, prepend=0)>1]+2],axis=1)))
    return amps