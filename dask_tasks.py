import numpy as np
from tqdm import trange
import dask.array as da
import cupy
import cusignal
from scipy.signal  import butter
import h5py
import time


def dask_tcs(recording, chunk_size=65536, overlap=1000, order=40, threshold=4, return_stds=False,  whitening=False):
    all_spikes = [[], []]
    spike_amps = []
    
    stds = np.std(recording.filtered_data(from_sample=0, to_sample=chunk_size), axis=1)
    for i in trange(round(recording.sample_length/chunk_size)):

        chunk = recording.filtered_data(from_sample=i*chunk_size, to_sample=(i+1)*chunk_size+overlap)
        chunk = da.array(chunk)
        if whitening:
            U, S, Vt = da.linalg.svd(chunk)
            chunk = cupy.dot(U.compute(), Vt.compute())
        chunk_spikes = cusignal.peak_finding.peak_finding.argrelmin(chunk, order=order, axis=1)
        spike_vals = chunk[chunk_spikes[0].get(), chunk_spikes[1].get()]
        sig_spikes = np.where(spike_vals <= - threshold*stds[chunk_spikes[0].get()])[0]
        all_spikes[0].append(chunk_spikes[0][sig_spikes])
        all_spikes[1].append(chunk_spikes[1][sig_spikes]+i*chunk_size)
        spike_amps.append(spike_vals[sig_spikes])
    all_spikes = cupy.array([cupy.concatenate(all_spikes[0]), cupy.concatenate(all_spikes[1])])
    if return_stds:
        return all_spikes, spike_amps, stds
    else:
        return all_spikes, spike_amps




def dask_filter_chunk(in_rec_filepath, channels, idx_from, idx_to, scales, low_cutoff, high_cutoff, order=3, cmr=True, whiten=True, h5write=None):
    sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')
    file = h5py.File(in_rec_filepath, 'r')
    sig = file['sig']
    chunk_size = idx_to - idx_from
    if idx_to > sig.shape[1]:
        idx_to = sig.shape[1]
        idx_from = idx_to - chunk_size
    if idx_from == 0:
        chunk = np.ones((len(channels), chunk_size*2))*sig[channels, 0][:, np.newaxis]
        chunk[:, chunk_size:] = sig[channels, :idx_to]
    else:
        chunk = sig[channels, idx_from-chunk_size:idx_to]
    saturations = np.count_nonzero(((0==chunk[:, chunk_size:]) | (chunk[:, chunk_size:] == 4095)), axis=1)
    file.close()
    cusig = cupy.asarray(chunk, dtype=cupy.float32)
    cusig = cusig - cupy.mean(cusig)
    cusig=cusignal.sosfilt(sos,cusig)
    cusig=cupy.fliplr(cusig)
    cusig=cusignal.sosfilt(sos,cusig)
    cusig=cupy.fliplr(cusig)
    cusig=cusig*cupy.asarray(scales, dtype=cupy.float32)[:,None]
    if cmr:
        cusig=cusig-cupy.median(cusig,axis=0)
    cusig = cusig.get()
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
        print('Completion:', len(np.where(np.array(status) == 'finished')[0])/len(status), end='\r')
        time.sleep(0.05)
        if any(np.array(status) == 'error') or any(np.array(status) == 'lost'):
            print('error on task')
            break