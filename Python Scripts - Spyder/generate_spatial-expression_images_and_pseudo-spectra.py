# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 18:56:59 2021

@author: nellin
"""

# import necessary modules
import numpy as np
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
from  matplotlib.patches import Patch
import matplotlib.image as mpimg
import pickle as pkl

# File names and folder paths to be used to generate image
# .sbd SCiLS file name
sbd_name = '2021-07-19_NE_MB_posMS_DHBsublim.sbd'

# Scores an loadings numpy array file names and paths
scores_file = 'PCA_2021-07-19_MB_DHB-Scores.npy'
scores_folder_path = 'Fingerprint_data/2021-07-19_NE_MB_posMS_DHBsublim/scores/'

loadings_file = "PCA_2021-07-19_MB_DHB-Loadings.npy"
loadings_folder_path = "Fingerprint_data/2021-07-19_NE_MB_posMS_DHBsublim/scores/"

# Save path for images
save_path = "Fingerprint_data/2021-07-19_NE_MB_posMS_DHBsublim/scores/"

# Path for folder containing .sbd file
sbd_folder_path = 'SBD_files/'

# Load all files
scores = np.load(f'{scores_folder_path}{scores_file}')
loadings = np.load(f'{loadings_folder_path}{loadings_file}')



# Import coordinates from SBD file
with h5py.File(f'{sbd_folder_path}{sbd_name}', mode = 'r') as h5:
    coords = h5["Registrations/0/Coordinates"][:]
    mz_range = h5["SamplePositions/GlobalMassAxis/SamplePositions"][:]
 
# Remap image coordinates
xcoords, ycoords = coords[0], coords[1]
new_xcoords, new_ycoords = xcoords.copy(), ycoords.copy()

for i, x in enumerate(np.unique(xcoords)):
    new_xcoords[new_xcoords == x] = i
for i, y in enumerate(np.unique(ycoords)):
    new_ycoords[new_ycoords == y] = -i-1

new_xcoords = new_xcoords.astype(np.int32)
new_ycoords = new_ycoords.astype(np.int32)

                   
# Generate Spatial expression images with whit background for each principal component
for pcx, PC in enumerate(scores):
    
    img = np.zeros([np.absolute(new_ycoords).max()+1, new_xcoords.max()+1])
    
    plt.figure(pcx, dpi=200)
    
    img_bckrnd = min(scores[pcx,:,1]) - 10
    
    img[:,:] = img_bckrnd
    
    for i, (y,x) in enumerate(zip(new_ycoords, new_xcoords)):
        img[y,x] = scores[pcx,i,1]
        
    bckrnd_coords = np.array(np.where(img == img_bckrnd))
    
    title = f"{scores_file.split('.')[0]}_PC{pcx+1}"
    
    plt.imsave(f'{save_path}{title}.jpg', img)
    
    img = plt.imread(f'{save_path}{title}.jpg').copy()
    
    img[bckrnd_coords[0,:], bckrnd_coords[1,:],:] = [255,255,255]
            
    img = img.repeat(30, axis=1).repeat(30, axis=0)
        
    plt.xticks([])
    plt.yticks([])
    mpimg.imsave(f'{save_path}{title}.jpg', img)
    plt.imshow(img)
    

# Generate pesudo-spectra for each principal component 
# creates one image with all plots 
# one for each principal component from top to bottom starting at principal component 1 
fig = plt.figure(figsize=(15,len(loadings)*3), dpi = 200)
gs = fig.add_gridspec(len(loadings), hspace = 0)
axs = gs.subplots(sharex = True)

title_loadings = f"{loadings_file.split('.')[0]}_pseudo-spectra"

for px, PC in enumerate(loadings):
    
    yloadings = np.zeros(len(mz_range))

    for mz,l in loadings[px,:,:]:
        abs_val = np.abs(mz_range - mz)
        closest_mzx = np.where(mz_range == mz_range[abs_val.argmin()])
        yloadings[closest_mzx] = l
        
    axs[px].plot(mz_range, yloadings, color = 'b')
    axs[px].set(ylabel=f'Loadings PC{px+1}')

xticks = np.linspace(round(min(mz_range)/10)*10, round(max(mz_range)/10)*10, 5)
plt.xticks(xticks)
plt.xlabel('m/z')
plt.savefig(f"{save_path}{title_loadings}", bbox_inches = 'tight')
plt.show()        
  

    
        