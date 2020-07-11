import cupy
import cusignal
from scipy.signal import butter
import numpy as np

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

def filter_traces(s, low_cutoff, high_cutoff, cmr=False):
    sos = butter(3, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')
    cusig=cupy.asarray(s, dtype=cupy.float32)
    cusig=cusig-cupy.mean(cusig)
    if cmr:
        cusig=cusig-cuda_median(cusig,0) 
    cusig=cusignal.sosfilt(sos,cusig)
    cusig=cupy.flipud(cusig)
    cusig=cusignal.sosfilt(sos,cusig)
    cusig=cupy.flipud(cusig)
    return cupy.asnumpy(cusig)

def filter_traces_batched(s, low_cutoff, high_cutoff, order=3, cmr=False, batch_size=1024):
    sos = butter(order, [low_cutoff/10000, high_cutoff/10000], 'bandpass', output='sos')
    n_chunks=s.shape[1]/batch_size
    chunks=np.hstack((np.arange(n_chunks, dtype=int)*batch_size,s.shape[1]))
    output=np.empty(s.shape)
    overlap=batch_size
    batch=np.zeros((s.shape[0],batch_size+overlap))
    batch[:,:overlap]=np.array([s[:,0],]*overlap).transpose()
    for i in range(len(chunks)-1):
        idx_from=chunks[i]
        idx_to=chunks[i+1]
        batch=batch[:,:(idx_to-idx_from+overlap)]
        batch[:,overlap:]=s[:,idx_from:idx_to]
        cusig=cupy.asarray(batch, dtype=cupy.float32)
        cusig=cusig-cupy.mean(cusig)
        if cmr:
            cusig=cusig-cuda_median(cusig,0) 
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.flipud(cusig)
        cusig=cusignal.sosfilt(sos,cusig)
        cusig=cupy.flipud(cusig)
        output[:,idx_from:idx_to]=cupy.asnumpy(cusig[:,overlap:])
        batch[:,:overlap]=batch[:,-overlap:]
    return output

def get_spike_amps(s):
    mean_stim_trace=cupy.asnumpy(cupy.mean(s,axis=0));
    spike_threshold=-cupy.std(mean_stim_trace)*4;
    crossings=np.where(mean_stim_trace<spike_threshold)[0][:-2];
    amps=np.abs(cupy.asnumpy(cupy.mean(s[:,crossings[np.diff(crossings, prepend=0)>1]+2],axis=1)))
    return np.vstack((np.where(amps>50),amps[amps>50]))    