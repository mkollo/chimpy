  
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
from tqdm import trange
import dask.array as da
try:
    import cupy
    import cusignal
except:
    pass
from scipy.signal  import butter
import h5py
import time


def dask_tcs(recording_filepath, idx_from, idx_to, chans, stds,  order=40, threshold=4,whitening=False):
    file = h5py.File(recording_filepath, 'r')
    if whitening:
        data = file['white_sig']
    else:
        data = file['sig']
#    chunk_size = idx_to - idx_to
    chunk = data[chans, idx_from:idx_to]
    file.close()
    chunk_spikes = cusignal.peak_finding.peak_finding.argrelmin(chunk, order=order, axis=1)
    spike_vals = chunk[chunk_spikes[0].get(), chunk_spikes[1].get()]
    sig_spikes = np.where(spike_vals <= - threshold*stds[chunk_spikes[0].get()])[0]
    return chunk_spikes[0][sig_spikes], chunk_spikes[1][sig_spikes]+idx_from, spike_vals[sig_spikes]

def dask_wf(recording_filepath, spike_time, channels, window=30, whitened=False):
    file = h5py.File(recording_filepath, 'r')
    data = file['sig'][channels, int(spike_time-window):int(spike_time+window)]
    if whitened:
        w_data = file['white_sig'][channels, int(spike_time-window):int(spike_time+window)]
    file.close()
    if whitened:
        return data, w_data
    else:
        return data

def dask_filter_chunk(in_rec_filepath, channels, idx_from, idx_to, scales, low_cutoff, high_cutoff, order=3, cmr=True, whiten=True, h5write=None):
    '''
    Filters a chunk of data from a h5py recording, designed to be used with dask
    '''

    # Generate filter
    sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')

    # Open the file and extract the sig
    file = h5py.File(in_rec_filepath, 'r')
    sig = file['sig']

    # Size of the chunk we're going to use. Take a chunk twice as big as this to allow for smooth filtering.
    chunk_size = idx_to - idx_from
    if idx_to > sig.shape[1]:
        idx_to = sig.shape[1]
        idx_from = idx_to - chunk_size
    if idx_from == 0: # Make a chunk of dc to prevent weird edge filtering problems
        chunk = np.ones((len(channels), chunk_size*2))*sig[channels, 0][:, np.newaxis]
        chunk[:, chunk_size:] = sig[channels, :idx_to]
    else:
        chunk = sig[channels, idx_from-chunk_size:idx_to]
    # Count saturations in the chunk
    saturations = np.count_nonzero(((0==chunk[:, chunk_size:]) | (chunk[:, chunk_size:] == 4095)), axis=1)
    file.close()
    # Filtering with CUDA
    cusig = cupy.asarray(chunk, dtype=cupy.float32)
    cusig = cusig - cupy.mean(cusig)
    cusig=cusignal.sosfilt(sos,cusig)
    cusig=cupy.fliplr(cusig)
    cusig=cusignal.sosfilt(sos,cusig)
    cusig=cupy.fliplr(cusig)
    cusig=cusig*cupy.asarray(scales, dtype=cupy.float32)[:,None]
    if cmr:
        cusig=cusig-cupy.median(cusig,axis=0)
    cusig = cusig.get() # Conver to numpy
    if whiten:
        U, S, Vt = np.linalg.svd(cusig, full_matrices=False)
        w_chunk = np.dot(U, Vt)
        if h5write is not None:
            written=False
            while not written:
                try:
                    out_fid = h5py.File(h5write, 'r+')
                    out_fid['sig'][:, idx_from:idx_to] = cusig[:, chunk_size:]
                    out_fid['white_sig'][:, idx_from:idx_to] = w_chunk[:, chunk_size:]
                    out_fid.close()
                    written = True
                except:
                    time.sleep(0.05)
            return saturations
        else:
            return cusig, w_chunk, saturations
    else:
        if h5write is not None:
            written=False
            while not written:
                try:
                    out_fid = h5py.File(h5write, 'r+')
                    if idx_from == 0:
                        out_fid['sig'][:, :idx_to] = cusig[:, :idx_to]
                    out_fid['sig'][:, int(idx_from+chunk_size/2):idx_to] = cusig[:, int(chunk_size/2):]
                    out_fid.close()
                    written = True
                except:
                    time.sleep(0.05)
            return saturations
        else:
            return cusig, saturations
    

def check_progress(dask_outs, progress_plot=False):
    status = [i.status for i in dask_outs]
    while len(np.where(np.array(status) == 'finished')[0]) != len(status):
        status = [i.status for i in dask_outs]
        print('Dask Completion:', len(np.where(np.array(status) == 'finished')[0])/len(status), end='\r')
        time.sleep(0.5)
        if any(np.array(status) == 'error') or any(np.array(status) == 'lost'):
            print('error on task')
            break
