#    ___ _  _ ___ __  __ _____   __
#   / __| || |_ _|  \/  | _ \ \ / /
#  | (__| __ || || |\/| |  _/\ V / 
#   \___|_||_|___|_|  |_|_|   |_| 
# 
# Copyright (c) 2020 Mihaly Kollo. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.

import dxchange.reader as dxreader
from skimage.feature import peak_local_max
from skimage import io, filters
import numpy as np
from tqdm import trange
from scipy.spatial.distance import cdist

class Ct():
    
    def __init__(self, ct_file):
        self.data, self.metadata = dxreader.read_txrm('/camp/home/kollom/working/mkollo/CHIME/BR_200710/ct/stitch-A/BR_200710_stitch-A_Stitch.txm')
        self.pixel_size=metadata['pixel_size']
        self.wire_pixels = int(50/self.pixel_size)
        
    def process(self):
        self.threshold()
        self.detect_wires()
        self.link_wires()

    def threshold(self):
        print("Thresholding images")
        self.midimage=self.data[self.data.shape[0]//2,:,:]
        self.threshold=filters.threshold_multiotsu(self.midimage)[-1]
        self.thresholded=self.data.copy()
        self.thresholded[self.data<self.threshold]=0

    def detect_wires(self):      
        coordinates=[]
        for i in trange(self.thresholded.shape[0], ncols=100, position=0, leave=True, desc="Detecting wire locations"):
            xy = peak_local_max(self.thresholded[i,:,:], min_distance=self.wire_pixels, threshold_abs=0.25)
            coord = np.zeros(shape=(xy.shape[0], 3))
            coord[:, :-1] = xy
            coord[:, 2] = i
            coordinates.append(coord)
        self.wire_locations = np.concatenate(coordinates)
          
    def find_coord_cluster(self, coord, clusters):
        for i in range(len(clusters)):
            if (clusters[i][-1, :] == coord).all():
                return i
        return None
    
    def link_wires(self):
        max_plane = int(max(self.wire_locations[:, 2]))
        min_plane = int(min(self.wire_locations[:, 2]))
        self.clusters = []
        coords_prev = self.wire_locations[self.wire_locations[:, 2] == max_plane, :]
        for i in range(coords_prev.shape[0]):
            self.clusters.append(coords_prev[i, :])
            self.clusters[-1] = np.expand_dims(self.clusters[-1], axis=0)
        for z in trange(max_plane - 1, 0, -1, ncols=100, position=0, leave=True, desc="Linking wires"):
            coords = self.wire_locations[self.wire_locations[:, 2] == z, :]
            distances = cdist(coords, coords_prev)
            distances_flat = distances.flatten()
            if distances_flat.shape[0] > 0:
                n = 0
                while True:
                    closest_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
                    n += 1
                    if distances[closest_index] == np.inf:
                        break
                    else:
                        closest_cluster = self.find_coord_cluster(coords_prev[closest_index[1]], self.clusters)
                        if (closest_cluster is not None) and (
                            distances[closest_index] <= np.sqrt(wire_pixels ^ 2 + 1) * 2
                        ):
                            self.clusters[closest_cluster] = np.vstack(
                                (self.clusters[closest_cluster], coords[closest_index[0]])
                            )
                            distances[closest_index[0], :] = np.inf
                            distances[:, closest_index[1]] = np.inf
                        else:
                            self.clusters.append(coords[closest_index[0]])
                            self.clusters[-1] = np.expand_dims(self.clusters[-1], axis=0)
                            distances[closest_index[0], :] = np.inf
                coords_prev = coords
    
