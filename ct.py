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
    
#     CODE DUMP 1 - latest


import dxchange
import dxchange.reader as dxreader
import numpy as np
from tqdm import trange
from scipy.spatial.distance import cdist
from scipy.ndimage.measurements import center_of_mass, label
from skimage.feature import peak_local_max

# Load data
print("Loadig data...")
data, metadata = dxreader.read_txrm('/camp/home/kollom/working/mkollo/CHIME/BR_200710/ct/stitch-A/BR_200710_stitch-A_Stitch.txm')
pixel_size=metadata['pixel_size']
wire_pixels = int(25/pixel_size)

# Find wire locations at middle plane
print("Detecting mid-plane wire locations...")
mid_z_plane=data.shape[0]//2+4
image=data[mid_z_plane,:,:]
midwire_locs = peak_local_max(image, min_distance=wire_pixels, threshold_abs=8500)
midwire_locs=midwire_locs[midwire_locs[:,0]<720]
midwire_locs=midwire_locs[midwire_locs[:,1]<800]
wires=np.full((data.shape[0],midwire_locs.shape[0],3), np.inf, dtype=np.float)
wires[mid_z_plane,:,:2]=midwire_locs
wires[mid_z_plane,:,2]=0
    
# Search down
for z_plane in trange(mid_z_plane-1,0,-1, desc="Searching down",position=0, leave=True):
    image=data[z_plane,:,:]
    if z_plane==mid_z_plane-1:
        pred_loc=wires[mid_z_plane,:,:2]
    else:
        pred_loc=wires[z_plane+1,:,:2]*2-wires[z_plane+2,:,:2]
    wire_locs = peak_local_max(image, min_distance=wire_pixels, threshold_abs=8500) 
    distances = cdist(wires[z_plane+1,:,:2], wire_locs)           
    while not np.isinf(distances).all():
        best_pair=np.unravel_index(np.argmin(distances),distances.shape)
        wires[z_plane,best_pair[0],:2]=wire_locs[best_pair[1],:]
        wires[z_plane,best_pair[0],2]=distances[best_pair]
        distances[best_pair[0],:]=np.inf
        distances[:,best_pair[1]]=np.inf
 
# Search up
for z_plane in trange(mid_z_plane,wires.shape[0],1, desc="Searching up",position=0, leave=True):
    image=data[z_plane,:,:]
    pred_loc=wires[z_plane-1,:,:2]*2-wires[z_plane-2,:,:2]
    wire_locs = peak_local_max(image, min_distance=wire_pixels, threshold_abs=7900) 
    distances = cdist(wires[z_plane-1,:,:2], wire_locs)           
    while not np.isinf(distances).all():
        best_pair=np.unravel_index(np.argmin(distances),distances.shape)
        wires[z_plane,best_pair[0],:2]=wire_locs[best_pair[1],:]
        wires[z_plane,best_pair[0],2]=distances[best_pair]
        distances[best_pair[0],:]=np.inf
        distances[:,best_pair[1]]=np.inf
#         if distances[best_pair]<5:
#             wires[z_plane,best_pair[0],:]=np.nan
 
    

# Plot plane and markings
import matplotlib.pyplot as plt
# clean_wires=wires.copy()
z_plane=40
wire_selection=1
fig=plt.figure(1, figsize=(10,10))
plt.imshow(data[z_plane,:,:])  
# wire_locs = peak_local_max(data[z_plane,:,:], min_distance=wire_pixels, threshold_abs=8200) 
# plt.scatter(wires[:,1],wires[:,0],s=0.25,c='r'

plt.scatter(clean_wires[z_plane,:,1],clean_wires[z_plane,:,0],s=0.25,c='r')
# plt.xlim(300,800)
# plt.ylim(1000,500)
plt.show()

# Fitting plane
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

N_POINTS = 10
TARGET_X_SLOPE = 2
TARGET_y_SLOPE = 3
TARGET_OFFSET  = 5
EXTENTS = 5
NOISE = 5

# create random data
xs = tops[:,0]
ys = tops[:,1]
zs = tops[:,2]

# plot raw data
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(xs, ys, zs, color='b')

# do fit
tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

print("solution:")
print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
# print("errors:")
# print(errors)
# print("residual:")
# print(residual)

# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1],30),
                  np.arange(ylim[0], ylim[1],30))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
ax.plot_wireframe(X,Y,Z, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()



# Project point on plane
origo=np.array([0,0,3115.671])
normal=np.array([0.009, 0.012, 1])
print(tops[0,:])
print(origo-np.dot(tops[0,:]-origo, normal)*normal)
    
#     CODE DUMP 2
        import dxchange
        import dxchange.reader as dxreader
        import numpy as np
        from tqdm import trange
        from scipy.spatial.distance import cdist

        # Load data
        print("Loadig data...")
        data, metadata = dxreader.read_txrm('/camp/home/kollom/working/mkollo/CHIME/BR_200710/ct/stitch-A/BR_200710_stitch-A_Stitch.txm')
        pixel_size=metadata['pixel_size']
        wire_pixels = int(25/pixel_size)

        # Find wire locations at middle plane
        print("Detecting mid-plane wire locations...")
        mid_z_plane=data.shape[0]//2+4
        image=data[mid_z_plane,:,:]
        midwire_locs = peak_local_max(image, min_distance=wire_pixels, threshold_abs=8200)
        midwire_locs=midwire_locs[midwire_locs[:,0]<720]
        midwire_locs=midwire_locs[midwire_locs[:,1]<800]
        wires=np.full((data.shape[0],midwire_locs.shape[0],3), 5000, dtype=np.float)
        wires[mid_z_plane,:,:2]=midwire_locs
        wires[mid_z_plane,:,2]=0

        # Search down
        for z_plane in trange(mid_z_plane-1,0,-1, desc="Searching down",position=0, leave=True):
            image=data[z_plane,:,:]
            if z_plane==mid_z_plane-1:
                pred_loc=wires[mid_z_plane,:,:2]
            else:
                pred_loc=wires[z_plane+1,:,:2]*2-wires[z_plane+2,:,:2]
            wire_locs = peak_local_max(image, min_distance=wire_pixels, threshold_abs=8200) 
            distances = cdist(pred_loc, wire_locs)           
            while not np.isinf(distances).all():
                best_pair=np.unravel_index(np.argmin(distances),distances.shape)
                wires[z_plane,best_pair[0],:2]=wire_locs[best_pair[1],:]
                wires[z_plane,best_pair[0],2]=distances[best_pair]
                distances[best_pair[0],:]=np.inf
                distances[:,best_pair[1]]=np.inf 

        # Search up
        for z_plane in trange(mid_z_plane,wires.shape[0],1, desc="Searching up",position=0, leave=True):
            image=data[z_plane,:,:]
            pred_loc=wires[z_plane-1,:,:2]*2-wires[z_plane-2,:,:2]
            wire_locs = peak_local_max(image, min_distance=wire_pixels, threshold_abs=8000) 
            distances = cdist(pred_loc, wire_locs)           
            while not np.isinf(distances).all():
                best_pair=np.unravel_index(np.argmin(distances),distances.shape)
                wires[z_plane,best_pair[0],:2]=wire_locs[best_pair[1],:]
                wires[z_plane,best_pair[0],2]=distances[best_pair]
                distances[best_pair[0],:]=np.inf
                distances[:,best_pair[1]]=np.inf 


        # Resect breaks
        x=np.where(wires[:,:,2]>3000)[0]
        y=np.where(wires[:,:,2]>3000)[1]
        wires[x,y,:]=np.nan
        clean_wires=wires.copy()
        for i in range(wires.shape[1]):
            speed=np.abs(np.diff(wires[:1500,i,0]))+np.abs(np.diff(wires[:1500,i,1]))
            speed=np.convolve(speed, np.ones((3,))/3, mode='valid')
            clean_wires[:np.max(np.where(speed>10)[0])+1,i,:]=np.nan
            speed=np.abs(np.diff(wires[1500:,i,0]))+np.abs(np.diff(wires[1500:,i,1]))
            if len(np.where(speed>10)[0])>0:
                clean_wires[np.max(np.where(speed>10)[0])-1:,i,:]=np.nan


        
        
        
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
    
