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
from pathlib import Path
import os
from tqdm import tqdm
import dask
from chimpy.dask_tasks import check_progress, dask_tcs, dask_wf
import sys
# sys.path.append('/home/camp/warnert/spikeextractors')
# sys.path.append('/home/camp/warnert/spikesorters')
# import spikesorters as ss
# import spikeextractors as se
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)


def write_probe_file(filename, coords, radius):
    with open(filename, "w") as fid:
        fid.write("total_nb_channels = "+str(coords.shape[0])+"\n")
        fid.write("radius = "+str(radius)+"\n")        
        fid.write("channel_groups = {\n")
        fid.write("\t1: {\n")
        fid.write("\t\t'channels': list(range("+str(coords.shape[0])+")),\n")
        fid.write("\t\t'graph': [],\n")
        fid.write("\t\t'geometry': {\n")
        for i in range(coords.shape[0]):
            fid.write("\t\t\t"+str(i)+":  [  "+', '.join([str(c) for c in coords[i,:]])+"],\n")
        fid.write("\t\t}\n")
        fid.write("\t}\n")
        fid.write("}\n")


def get_default_kilosort_params():
    default_params = {
        'sample_rate': 20000,
        'kilo_thresh': 6,
        'projection_threshold': [10, 4],
        'preclust_threshold': 8,
        'car': 1,
        'minFR': 0.1,
        'minfr_goodchannels': 0.1,
        'freq_min': 150,
        'sigmaMask': 30,
        'nPCs': 3,
        'ntbuff': 64,
        'nfilt_factor': 4,
        'NT': 64 * 2,
        'keep_good_only': False,
        'chunk_mb': 500,
        'n_jobs_bin': 1,
        'kilosort2_path':'/home/camp/warnert/Kilosort2',
        'npymatlab_path':'/home/camp/warnert/npy-matlab/npy-matlab',
        'nchan':1028
        }
    return default_params

def write_kilosort_files(experiment, **kwargs):
    source_dir = Path(Path(__file__).parent)
    p = get_default_kilosort_params()
    for i in kwargs:
        p[i]=kwargs[i]
    connected_chans = np.zeros(p['nchan'])
    connected_chans[experiment.connected_pixels] = 1
    channel_map_text = open(os.path.join(source_dir, 'kilosort_templates', 'kilosort2_channelmap.m'), 'r').read()
    channel_map_text = channel_map_text.format(
        nchan=p['nchan'],
        connected=connected_chans,
        xcoords=experiment.recordings['brain'].estimated_coordinates[:, 0],
        ycoords=experiment.recordings['brain'].estimated_coordinates[:, 1],
        kcoords=np.ones(len(experiment.connected_pixels)),
        sample_rate=p['sample_rate']
    )
    config_text = open(os.path.join(source_dir, 'kilosort_templates', 'kilosort2_config.m'), 'r').read()
    config_text = config_text.format(
        nchan=p['nchan'],
        sample_rate=p['sample_rate'],
        main_file=experiment.recordings['brain'].filepath,
        freq_min=p['freq_min'],
        minfr_goodchannels=p['minfr_goodchannels'],
        projection_threshold=p['projection_threshold'],
        minFR=p["minFR"],
        sigmaMask=p['sigmaMask'],
        preclust_threshold=p['preclust_threshold'],
        kilo_thresh=p['kilo_thresh'],
        use_car=p['car'],
        nfilt_factor=p['nfilt_factor'],
        ntbuff=p['ntbuff'],
        NT=p['NT'],
        nPCs=p["nPCs"]
    )
    master_text = open(os.path.join(source_dir, 'kilosort_templates', 'kilosort2_master.m'), 'r').read()
    master_text = master_text.format(
        kilosort2_path=p['kilosort2_path'],
        npymatlab_path=p['npymatlab_path'],
        output_folder=experiment.savepath,
        channel_path=os.path.join(experiment.savepath, 'kilosort2_channelmap.m'),
        config_path=os.path.join(experiment.savepath, 'kilosort2_config.m'),
    )
    for fname, txt in zip(['kilosort2_master.m', 'kilosort2_config.m',
                               'kilosort2_channelmap.m'],
                              [master_text, config_text,
                               channel_map_text]):
            with open(os.path.join(experiment.savepath, fname), 'w') as f:
                f.write(txt)



# def run_ms4(recording_path, out_path, **params):
#     chimpy_rec = se.CHIMERecordingExtractor(recording_path)
#     chimpy_rec.read_chimpy_chan_params()
#     default_params = ss.Mountainsort4Sorter.default_params()
#     for i in params:
#         default_params[i] = params[i]
#     ms4_out = ss.run_mountainsort4(recording=chimpy_rec, output_folder=out_path, verbose=True, **params)
#     return ms4_out


def find_tcs(recording, chunk_size, client, order=40, threshold=4, whitening=False, return_stds=True):    
    try:
        stds = np.std(recording.filtered_data(from_sample=0, to_sample=chunk_size), axis=1)

        chunk_times = np.arange(0, recording.sample_length, chunk_size)
        tcs = [dask.delayed(dask_tcs)(recording.filtered_filepath, chunk_times[i], chunk_times[i+1], recording.channels, stds, order=order, threshold=threshold, whitening=whitening) for i in range(len(chunk_times)-1)]
        tcs_out = client.compute(tcs)

        check_progress(tcs_out)

        tc_times, tc_chans, tc_amps = [], [], []
        for i in tcs_out:
            tc_times.append(i.result()[0])
            tc_chans.append(i.result()[1])
            tc_amps.append(i.result()[2])

        tc_times = np.concatenate(tc_times)
        tc_chans = np.concatenate(tc_chans)
        tc_amps = np.concatenate(tc_amps)

        chan_tcs = [tc_chans[tc_times == i] for i in recording.channels]
        chan_amps = [tc_amps[tc_times.get()  == i] for i in recording.channels]

        if return_stds:
            return chan_tcs, chan_amps, stds
        else:
            return chan_tcs, chan_amps
    except KeyboardInterrupt:
        print('Cancelling jobs')
        [i.cancel() for i in tqdm(tcs)]
        


def find_wfs(recording, spike_times, client, window=30, whitened=False):
    try:
        wfs_delayed = [dask.delayed(dask_wf)(recording.filtered_filepath, i, recording.channels, whitened=whitened) for i in spike_times]

        wfs_out = client.compute(wfs_delayed)

        check_progress(wfs_out)
        wfs = [i.result() for i in tqdm(wfs_out)]
        return wfs
    except KeyboardInterrupt:
        print('Cancelling jobs')
        [i.cancel() for i in tqdm(tcs)]