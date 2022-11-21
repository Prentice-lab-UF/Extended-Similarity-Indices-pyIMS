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
sbd_name = '2021-07-27_NE_MB_posMS_DHBsublim.sbd'
sbd_folder_path = 'SBD_files/'

selected_pixels_file = 'PCA_2021-07-27_MB_DHB-Scores_sp08'
selected_pixels_folder_path = 'Fingerprint_data/2021-07-27_NE_MB_posMS_DHBsublim/selected_pixels/'

scores_file = 'PCA_2021-07-27_MB_DHB-Scores.npy'
scores_folder_path = 'Fingerprint_data/2021-07-27_NE_MB_posMS_DHBsublim/scores/'

medoid_pixels_file = '2021-07-27_NE_MB_posMS_DHBsublim_local_it11_MB_DHB-Scores_sp08_medoid-spectra'
medoid_pixels_folder_path = 'Results/2021-07-27_NE_MB_posMS_DHBsublim/RR_medoid/'

save_path = "Results/2021-07-27_NE_MB_posMS_DHBsublim/plots/selected_pixels_img/test/"



# Load all files
selected_pixels = pkl.load(open(f'{selected_pixels_folder_path}{selected_pixels_file}', "rb"))
scores = np.load(f'{scores_folder_path}{scores_file}')
medoid_pixels = pkl.load(open(f'{medoid_pixels_folder_path}{medoid_pixels_file}', "rb"))

# Import coordinates from SBD file
with h5py.File(f'{sbd_folder_path}{sbd_name}', mode = 'r') as h5:
    coords = h5["Registrations/0/Coordinates"][:]
    
# Remap image coordinates
xcoords, ycoords = coords[0], coords[1]
new_xcoords, new_ycoords = xcoords.copy(), ycoords.copy()

for i, x in enumerate(np.unique(xcoords)):
    new_xcoords[new_xcoords == x] = i
for i, y in enumerate(np.unique(ycoords)):
    new_ycoords[new_ycoords == y] = -i-1

new_xcoords = new_xcoords.astype(np.int32)
new_ycoords = new_ycoords.astype(np.int32)


# legend handle for image
legend_elements = [Patch(facecolor='red', edgecolor='black', label='Low scores'), 
                   Patch(facecolor='purple', edgecolor='black', label='Mid scores'),
                   Patch(facecolor='blue', edgecolor='black', label='High scores'),
                   Patch(facecolor='white', edgecolor='black', label='Low medoid'),
                   Patch(facecolor='pink', edgecolor='black', label='Mid medoid'),
                   Patch(facecolor='cyan', edgecolor='black', label='High medoid')]
                   
# save legend as individual image
legendfig = plt.figure("Legend Plot", dpi=800)
legendfig.legend(handles=legend_elements, loc='center')
legendfig.savefig(f"{save_path}PC_regions_legend.jpg")

# Generate Image for each principal component to show three regions of 
# interest based on scores values and most representative spectra locations for 
# each region
for pcx, PC in enumerate(selected_pixels):
    
    # Create image matrix based on remapped coords
    img = np.zeros([np.absolute(new_ycoords).max()+1, new_xcoords.max()+1])
    
    # Create figure
    plt.figure(PC, dpi=500)
    
    # Define image background as value outside range of scores values
    img_bckrnd = min(scores[pcx,:,1]) - 10
    
    # Set image matrix to background 
    img[:,:] = img_bckrnd
    
    # Create spatial-expression images (scores images for PC)
    for i, (y,x) in enumerate(zip(new_ycoords, new_xcoords)):
        img[y,x] = scores[pcx,i,1]
        
    # Find coords for background of spatial-expression images
    bckrnd_coords = np.array(np.where(img == img_bckrnd))
    
    # Set title and save image matrix as .jpg (must be saved as .jpg here so colors can be manipulated properly)
    # This image is the spatial-expression image for the PC 
    title = medoid_pixels_file.split("_")
    title = "_".join(title[0:-1])+f'_{PC}'
    plt.imsave(f'{save_path}{title}_regions_image.jpg', img)
    
    # Open .jpg file
    img = plt.imread(f'{save_path}{title}_regions_image.jpg').copy()
    
    # Set background color of .jpg image to black
    img[bckrnd_coords[0,:], bckrnd_coords[1,:],:] = [255,255,255]
    
    # Define the selected pixels region colors and the medoid spectra colors 
    # Colors must be in RGB format since .jpg files are composed of RGB pixels 
    r_colors = [[255,0,0], [128, 0, 128], [0,0, 255]]    
    m_colors = [[255,255,255], [255, 192, 203], [0, 255, 255]]
    
    # Set the selected pixels to their respective colors 
    for rx, region in enumerate(selected_pixels[PC]):
        for p in region:
            x, y = new_xcoords[p], new_ycoords[p]
            img[y,x,:] = r_colors[rx]
    
    # Set the medoid pixels to their respective color
    for mx, mp in enumerate(medoid_pixels[PC]):
        x, y = new_xcoords[mp], new_ycoords[mp]
        img[y,x,:] = m_colors[mx]
        
    # Save .jpg image with regions and medoid spectra 
    plt.title(f"{title}_regions_image")
    plt.xticks([])
    plt.yticks([])
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.65,1))
    mpimg.imsave(f'{save_path}{title}_regions_image.jpg', img.repeat(25, axis=1).repeat(25, axis=0))
    plt.imshow(img)
    
        