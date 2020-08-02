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
import h5py
from chimpy.slurm import Slurm
from chimpy.ramfile import RamFile

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
    n_sample_chunks=n_samples/sample_chunk_size
    chunks=np.hstack((np.arange(n_sample_chunks, dtype=int)*sample_chunk_size,n_samples))
    output=np.empty((s.shape[0], n_samples))
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

def filter_experiment_slurm(exp, low_cutoff, high_cutoff, order=3, cmr=False, sample_chunk_size=65536, n_samples=-1, ram_copy=False):
    #     Optionally save file into a tmpfs partition for processing
    stim_recording=exp.recordings['stim']
    brain_recording=exp.recordings['brain']
    channels=stim_recording.channels
    amps=stim_recording.amps
    scales=1000/amps
    in_filepath=brain_recording.filepath
    out_filepath=brain_recording.filtered_filepath
    params={
        'sample_chunk_size':sample_chunk_size,
        'channels':list(map(int, channels)),
        'n_samples':n_samples,
        'cmr':cmr,
        'order':order,
        'low_cutoff':low_cutoff,
        'high_cutoff':high_cutoff,
        'scales':list(map(float,scales)),
        'in_filepath': in_filepath,
        'out_filepath': out_filepath,
        'connected_pixels': list(map(int, exp.connected_pixels)),
        'ram_copy': ram_copy
    }
    slurm = Slurm("slurm_filter", 12, gpu=True)
    slurm.run(params)

def filter_experiment_local(in_recording, stim_recording, low_cutoff, high_cutoff, order=3, cmr=False, sample_chunk_size=65536, n_samples=-1, ram_copy=False):    
    channels=stim_recording.channels
    amps=stim_recording.amps
    scales=1000/amps
    n_channels=stim_recording.channels.shape[0]
#     Optionally save file into a tmpfs partition for processing
    if ram_copy:
        in_ramfile=RamFile(in_recording.filepath, 'r')
        in_filepath=in_ramfile.ram_filepath
        out_ramfile=RamFile(in_recording.filtered_filepath, 'w')
        out_filepath=out_ramfile.ram_filepath
    else:
        in_filepath=in_recording.filepath
        out_filepath=in_recording.filtered_filepath
    in_fid=h5py.File(in_filepath, 'r')
#     Create output file
    out_fid=h5py.File(out_filepath, 'w')
    if n_samples==-1:
        n_samples=in_fid['sig'].shape[1]
    out_fid['mapping']=in_fid['mapping'][stim_recording.connected_in_mapping]
    in_fid.copy('/message_0', out_fid)
    in_fid.copy('/proc0', out_fid)
    in_fid.copy('/settings', out_fid)
    in_fid.copy('/time', out_fid)
    in_fid.copy('/version', out_fid)
    if 'bits' in in_fid.keys():
        in_fid.copy('/bits', out_fid)
    out_fid.create_dataset("sig", (n_channels, n_samples), dtype='float32')
#     Create filter: cutoff / 0.5 * fs
    sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')
#     Create chunks
    n_sample_chunks=n_samples/sample_chunk_size
    sample_chunks=np.hstack((np.arange(n_sample_chunks, dtype=int)*sample_chunk_size,n_samples))
    overlap=sample_chunk_size
    chunk=np.zeros((n_channels,sample_chunk_size+overlap))
    chunk[:,:overlap]=np.array([in_fid['sig'][channels,0],]*overlap).transpose()
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

def get_spike_crossings(s):
    mean_stim_trace=cupy.asnumpy(cupy.mean(s,axis=0));
    spike_threshold=-cupy.std(mean_stim_trace)*7;
    crossings=np.where(mean_stim_trace<spike_threshold)[0][:-2];
    return crossings[np.diff(crossings, prepend=0)>1]
    
def get_spike_amps(s):
    sig=cupy.asarray(s[:1024,:20000])
    peaks=cusignal.peak_finding.peak_finding.argrelmin(sig, order=20, axis=1)
    mean_std=cupy.mean(cupy.std(sig,axis=1))
    significant_peaks=sig[peaks[0],peaks[1]]<(-5*mean_std)
    amps=np.median(cupy.asnumpy(sig[:,peaks[1][significant_peaks]]*-1),axis=1)
    return amps

def power_spectrum(data):
    fourier_transform = np.fft.rfft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    power_spectrum_smoothed=np.convolve(power_spectrum, np.ones((20,))/20, mode='valid')
    frequency = np.linspace(0, 10000, len(power_spectrum_smoothed))
    return frequency, power_spectrum_smoothed
