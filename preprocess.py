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
cuda_present = False
try:
    import cupy
    import cusignal
    cuda_present = True
except:
    print('Unable to import cuda pacakges, don\'t try and functions which need them')
from scipy.signal import butter
import numpy as np
from tqdm import trange, tqdm
import re
import importlib
import h5py
from chimpy.slurm import Slurm
from chimpy.ramfile import RamFile
from chimpy.dask_tasks import dask_filter_chunk, check_progress
import dask

def cuda_median(a, axis=1):
    assert cuda_present, "Cuda isn't implemented, cannot run function"
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
    assert cuda_present, "Cuda isn't implemented, cannot run function"
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
        cusig=cupy.fliplr(cusig)
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.fliplr(cusig)
        output[:,idx_from:idx_to]=cupy.asnumpy(cusig[:,overlap:])
        chunk[:,:overlap]=chunk[:,-overlap:]
    return output



def filter_dask(in_recording, stim_recording, low_cutoff, high_cutoff, client, order=3, sample_chunk_size=65536, n_samples=-1, cmr=True, whiten=True, dask_write=False, show_progress=False):
    
    # Set up the parameters from the recordings
    channels=stim_recording.channels
    amps=stim_recording.amps
    scales=1000/amps
    n_channels=stim_recording.channels.shape[0]
    # Optionally save file into a tmpfs partition for processing
    in_filepath=in_recording.filepath
    out_filepath=in_recording.filtered_filepath

    # Open the input
    in_fid=h5py.File(in_filepath, 'r')
    # Create output file
    out_fid=h5py.File(out_filepath, 'w')
    if n_samples==-1:
        n_samples=in_fid['sig'].shape[1]
    out_mapping=in_fid['mapping'][stim_recording.connected_in_mapping]
    for i, m in enumerate(out_mapping):
        m[0]=i    
    out_fid['mapping']=out_mapping

    # Copy over the parameters over
    in_fid.copy('/message_0', out_fid)
    in_fid.copy('/proc0', out_fid)
    in_fid.copy('/settings', out_fid)
    in_fid.copy('/time', out_fid)
    in_fid.copy('/version', out_fid)
    if 'bits' in in_fid.keys():
        in_fid.copy('/bits', out_fid)

    # New dataset for the filtered data and whitened if whitening is True
    out_fid.create_dataset("sig", (n_channels, n_samples), dtype='float32')
    if whiten:
        out_fid.create_dataset('white_sig', (n_channels, n_samples), dtype='float32')
    # Create chunks

    # Create the number of chunks
    n_sample_chunks=n_samples/sample_chunk_size
    sample_chunks=np.hstack((np.arange(n_sample_chunks, dtype=int)*sample_chunk_size,n_samples))

    # Dataset for saturations and the first frame
    out_fid.create_dataset('saturations', (n_channels, len(sample_chunks-1)), dtype='int32')
    out_fid.create_dataset('first_frame', shape=(1,), data=in_fid["sig"][1027,0]<<16 | in_fid["sig"][1026,0])

    # Create a series of dask delayed to do the filtering
    h5write = None
    if dask_write:
        h5write = out_filepath
    filters = [dask.delayed(dask_filter_chunk)(in_recording.filepath, channels, sample_chunks[i], sample_chunks[i+1], scales, low_cutoff, high_cutoff,order=order, cmr=cmr, whiten=whiten, h5write=h5write) for i in range(len(sample_chunks)-1)]

    # Close the out fid if the dask delayed will write to the output
    if dask_write:
        out_fid.close()

    # Compute and show progress
    filters_out = client.compute(filters)
    if show_progress:
        check_progress(filters_out)
    
    # Write to the out_fid if the dask tasks aren't doing it
    out_fid=h5py.File(out_filepath, 'a')
    if not dask_write:
        if whiten:
            for i in trange(len(sample_chunks)-1):
                out_fid['white_sig'][:, sample_chunks[i]:sample_chunks[i+1]] = filters_out[i].result()[1][:, sample_chunk_size:]
        for i in trange(len(sample_chunks)-1):
            out_fid['sig'][:, sample_chunks[i]:sample_chunks[i+1]] = filters_out[i].result()[0][:, sample_chunk_size:]
            out_fid['saturations'][:, i] = filters_out[i].result()[-1]
    else:
        for i in trange(len(sample_chunks)-1):
            out_fid['saturations'][:, i] = filters_out[i].result()
    out_fid.close()
    return filters_out

def filter_experiment_slurm(in_recording, stim_recording, low_cutoff, high_cutoff, order=3, sample_chunk_size=65536, n_samples=-1, ram_copy=False):
    #     Optionally save file into a tmpfs partition for processing   
    channels=stim_recording.channels
    amps=stim_recording.amps
    scales=1000/amps
    in_filepath=in_recording.filepath
    out_filepath=in_recording.filtered_filepath
    params={
        'sample_chunk_size':sample_chunk_size,
        'channels':list(map(int, channels)),
        'n_samples':n_samples,
        'order':order,
        'low_cutoff':low_cutoff,
        'high_cutoff':high_cutoff,
        'scales':list(map(float,scales)),
        'in_filepath': in_filepath,
        'out_filepath': out_filepath,
        'connected_pixels': list(map(int, stim_recording.connected_pixels)),
        'ram_copy': ram_copy
    }
    slurm = Slurm("slurm_filter", 12, gpu=True)
    slurm.run(params)

def filter_experiment_local(in_recording, stim_recording, low_cutoff, high_cutoff, order=3, cmr=False, sample_chunk_size=65536, n_samples=-1, ram_copy=False, whiten=False):
    assert cuda_present, "Cuda isn't implemented, cannot run function" 
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
    out_mapping=in_fid['mapping'][stim_recording.connected_in_mapping]
    for i, m in enumerate(out_mapping):
        m[0]=i    
    out_fid['mapping']=out_mapping
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
    out_fid.create_dataset('saturations', (n_channels, len(sample_chunks-1)), dtype='int32')
    out_fid.create_dataset('first_frame', shape=(1,), data=in_fid["sig"][1027,0]<<16 | in_fid["sig"][1026,0])
    overlap=sample_chunk_size
    chunk=np.zeros((n_channels,sample_chunk_size+overlap))
    chunk[:,:overlap]=np.array([in_fid['sig'][channels,0],]*overlap).transpose()
    for i in trange(len(sample_chunks)-1, ncols=100, position=0, leave=True):
        idx_from=sample_chunks[i]
        idx_to=sample_chunks[i+1]
        chunk=chunk[:,:(idx_to-idx_from+overlap)]
        chunk[:,overlap:]=in_fid['sig'][channels,idx_from:idx_to]
        out_fid['saturations'][:, i] = np.count_nonzero(((0==chunk[:, overlap:]) | (chunk[:, overlap:] == 4095)), axis=1)
        cusig=cupy.asarray(chunk, dtype=cupy.float32)
        cusig=cusig-cupy.mean(cusig) 
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.fliplr(cusig)
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.fliplr(cusig)
        cusig=cusig*cupy.asarray(scales, dtype=cupy.float32)[:,None]
        if cmr:
            cusig=cusig-cupy.median(cusig,axis=0)
        out_fid["sig"][:,idx_from:idx_to]=cupy.asnumpy(cusig[:,overlap:])
        chunk[:,:overlap]=chunk[:,-overlap:]
#     Writing filtered traces to disk...
    in_fid.close()
    out_fid.close()
    if ram_copy:
        in_ramfile.save()
        out_ramfile.save()
        del in_ramfile, out_ramfile

def get_spike_crossings(s, threshold=7):
    assert cuda_present, "Cuda isn't implemented, cannot run function"
    mean_stim_trace=cupy.asnumpy(cupy.mean(s,axis=0))
    spike_threshold=-cupy.std(mean_stim_trace)*threshold
    crossings=np.where(mean_stim_trace<spike_threshold)[0][:-2]
    return crossings[np.diff(crossings, prepend=0)>1]
    
def get_spike_amps(s, return_counts = False):
    assert cuda_present, "Cuda isn't implemented, cannot run function"
    sig=cupy.asarray(s[:1024,:20000])
    peaks=cusignal.peak_finding.peak_finding.argrelmin(sig, order=20, axis=1)
    mean_std=cupy.mean(cupy.std(sig,axis=1))
    significant_peaks=sig[peaks[0],peaks[1]]<(-10*mean_std)
    amps=np.median(cupy.asnumpy(sig[:,peaks[1][significant_peaks]]*-1),axis=1)
    if return_counts:
        sig_peak_chans = peaks[0][significant_peaks]
        chan_count = np.array([len(sig_peak_chans[sig_peak_chans == i]) for i in range(1024)])
        return amps, chan_count
    else:
        return amps

def power_spectrum(data):
    fourier_transform = np.fft.rfft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    power_spectrum_smoothed=np.convolve(power_spectrum, np.ones((20,))/20, mode='valid')
    frequency = np.linspace(0, 10000, len(power_spectrum_smoothed))
    return frequency, power_spectrum_smoothed

