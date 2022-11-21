# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:44:09 2021

@author: nellin
"""

# Import necessary modules
import numpy as np
import pickle as pkl
import h5py
import hdf5plugin 
import matplotlib.pyplot as plt
import os

# Define normalization function to normalize MS datasets from SCiLs_raw_intensities() function
def normalize(raw_spc, function = "RMS"):
    
    # Function description
    """
    
    Arguments
    ----------
    raw_spc : array from SCiLs_raw_intensities() function output
        DESCRIPTION : 2D array where each row is a spectrum and each column is an m/z bin (n spectra, m bins)
            Recommended to use the raw output from the SCiLs_raw_intensities() function
            
    function : 'str'
        DESCRIPTION : define the normalization method to be used. Options are only 'RMS'(Root Mean Square) or 'TIC' (Total Ion Count)
    
    Returns
    -------
    norm_spc : 2D array 
        DESCRIPTION : 2D data matrix containing the normalized intensities.
        format : Data matrix where each row makes up the intensities for a single spectrum and each column  is a single m/z bin. Shape of the matrix is in the following format (n spectra, m bins).
       
    """
    
    #Import necessary modules
    import numpy as np
    
    # Creat norm_spc array and set number of bins
    norm_spc = raw_spc
    l = len(raw_spc.transpose())
    
    # Calculate normalized intensities based on the two methods available
    if function == "RMS":
        for i, pixel in enumerate(norm_spc):
            sum_squared = np.sum(pixel**2)
            mean_square = sum_squared/l
            RMS = mean_square**(1/2)
            pixel/=RMS
    if function == "TIC":
        for i, pixel in enumerate(norm_spc):
            TIC = np.sum(pixel)
            pixel/=TIC 
            
    # Return the normalized spectra
    return norm_spc

# File names and folder paths to be used to generate image
sbd_name = "2021-07-27_NE_MB_posMS_DHBsublim.sbd"

normalization = None

medoid_file = '2021-07-27_NE_MB_posMS_DHBsublim_local_it11_MB_DHB-Scores_sp08_medoid-spectra'
medoid_folder = 'Results/2021-07-27_NE_MB_posMS_DHBsublim/RR_medoid/' 

# PCA loadings file (must be numpy format not teh CSV export form SCiLS)
loadings_file = 'PCA_2021-07-27_MB_DHB-Loadings.npy'
loadings_folder = 'Fingerprint_data/2021-07-27_NE_MB_posMS_DHBsublim/scores/'

total_fp_file = '2021-07-27_NE_MB_posMS_DHBsublim_local_it11.npy'
total_fp_folder = 'Fingerprint_data/2021-07-27_NE_MB_posMS_DHBsublim/total_fingerprints/'

save_path = 'Results/2021-07-27_NE_MB_posMS_DHBsublim/plots/medoid_spectra/test/'


# Load all files
medoid_indices = pkl.load(open(f'{medoid_folder}{medoid_file}', "rb"))
loadings = np.load(f"{loadings_folder}{loadings_file}")
total_fp = np.load(f"{total_fp_folder}{total_fp_file}")
total_fp = np.array(total_fp, dtype = 'int8')
sbd_folder = 'SBD_files/'

# Import raw intensities and m/z's from SCiLS software
with h5py.File(f'{sbd_folder}{sbd_name}', mode="r") as h5:
    total_spectra = h5["SpectraGroups/InitialMeasurement/spectra"][:]
    
    mz_range = h5["SamplePositions/GlobalMassAxis/SamplePositions"][:]

total_spectra = normalize(total_spectra, function = normalization)

# Pull intensity threshold from total_fp_file 
it = float(total_fp_file.split(".")[0].split("_")[-1].replace("it", "."))

# Create region list labels
r = ['low', 'mid', 'high']

# Iterate across the medoid pixels dictionary 
for px, PC in enumerate(medoid_indices):
    
    # Iterate across the each PC within the medoid dictionary
    for rx,region in enumerate(medoid_indices[PC]):
        
        # Create figure
        fig = plt.figure(figsize = (15,15), dpi = 200)
        gs = fig.add_gridspec(len(region)+1, hspace = 0)
        axs = gs.subplots(sharex = True)
        
        # Set title 
        title = f'{medoid_file}_{PC}_{r[rx]}'
        
        # Plot each medoid spectra from the selected pixels region 
        # Multiple medoid specta for each region can result so it is necessary to look at all of them 
        for spx, spectrum in enumerate(region):
            
            # Plot raw spectra in blue
            axs[spx].plot(mz_range, total_spectra[spectrum], color = 'b')
            
            # Replot the peaks that were assigned as 1's in the binary fingerprint in red
            axs[spx].plot(mz_range, (total_fp[spectrum]*total_spectra[spectrum]), color = 'r')
            
            # Clean up baseline of spectra 
            axs[spx].plot(mz_range, np.zeros(len(mz_range)), color = 'b')
            
            # If intensity threshold is non-zero plot a grey line to show where the intensity threshold is located
            axs[spx].plot(mz_range, (np.max(total_spectra[spectrum])*it).repeat(len(mz_range)), color = 'grey')
            
            axs[spx].set(ylabel='intensity (a.u.)')
        
        # Ploy baseline of pseudo-spectra (loadings of PCA)
        yloadings = np.zeros(len(mz_range))
        
        # Create loadings pseudot-spectra y axis
        for mz,l in loadings[px,:,:]:
            abs_val = np.abs(mz_range - mz)
            closest_mzx = np.where(mz_range == mz_range[abs_val.argmin()])
            yloadings[closest_mzx] = l
        
        # Plot pseudo-spectra
        axs[len(region)].plot(mz_range, yloadings, color = 'b')
        axs[len(region)].set_title('pseudo-spectra', y = 1, pad = -25)
        axs[len(region)].set(ylabel='intensity (a.u.)')
        
        # Merge spectra to share x-axis
        for ax in axs:
            ax.label_outer()
        
        # Label spectra and save 
        fig.suptitle(title)
        fig.tight_layout()
        plt.xlabel('m/z')
        plt.savefig(f"{save_path}{title}", bbox_inches = 'tight')
        plt.show()        