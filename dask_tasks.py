import numpy as np
from tqdm import trange
import dask.array as da
import cupy as cp
import cusignal

def dask_tcs(recording, chunk_size=65536, overlap=1000, order=40, threshold=4, return_stds=False,  whitening=False):
    all_spikes = [[], []]
    spike_amps = []
    stds = np.std(recording.filtered_data(from_sample=0, to_sample=chunk_size), axis=1)
    for i in trange(round(recording.sample_length/chunk_size)):

        chunk = recording.filtered_data(from_sample=i*chunk_size, to_sample=(i+1)*chunk_size+overlap)
        chunk = da.array(chunk)
        if whitening:
            U, S, Vt = da.linalg.svd(chunk)
            chunk = cp.dot(U.compute(), Vt.compute())
        chunk_spikes = cusignal.peak_finding.peak_finding.argrelmin(chunk, order=order, axis=1)
        spike_vals = chunk[chunk_spikes[0].get(), chunk_spikes[1].get()]
        sig_spikes = np.where(spike_vals <= - threshold*stds[chunk_spikes[0].get()])[0]
        all_spikes[0].append(chunk_spikes[0][sig_spikes])
        all_spikes[1].append(chunk_spikes[1][sig_spikes]+i*chunk_size)
        spike_amps.append(spike_vals[sig_spikes])
    all_spikes = cp.array([cp.concatenate(all_spikes[0]), cp.concatenate(all_spikes[1])])
    if return_stds:
        return all_spikes, spike_amps, stds
    else:
        return all_spikes, spike_amps

    