# Import necessary modules
import numpy as np
import h5py
import hdf5plugin # optional
import matplotlib.pyplot as plt

def normalize(raw_spc, function = "RMS"):
    norm_spc = raw_spc
    l = len(raw_spc.transpose())
    
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
    return norm_spc


# Define name of .sbd file to extract spectrum data
name = "2021-07-27_NE_MB_posMS_DHBsublim.sbd"

# Define folder path where .sbd file is stored
folder = 'SBD_files/'

# Define save path where the binary data set will be stored and saved
save_path = 'Fingerprint_data/2021-07-27_NE_MB_posMS_DHBsublim/total_fingerprints/'


# Choose pre-processing normalization (if normalization was done for PCA) and 0-1 normalization type to be used to convert raw spectra into binary fingerprints
# Options for pre-processing normalization:
    # RMS: normalizes each spectrum to the root mean square of the spectrum
    # Total Ion Count: normalizes each spectra to the total ion count  (redundant for to use this and localTIC as they are the same)
    # None: No pre-processing normalization is done
# Options for 0-1 normalization:
    # global: finds the largest intensity of the data set and divides all other intensities by that value
    # local: for each pixel/spectrum, each intensity is divided by the largest intensity in the single spectrum
    # localTIC: for each pixel/spectrum, all intensities in the spectrum are added and then each intensity is divided by the sum
    # globalTIC: finds the sum of intenisties for every spectrum and divides all intensity values by the largest sum
    # PCA: Normalize each mz bin with repect to the average intensity of the bin (0-centered data used for PCA calculation)
pre_norm = None
normalization = "local"

# Using h5py moule open .sbd file and extract all raw spectra from imaging data set defined as raw_arr
with h5py.File(f'{folder}{name}', mode="r") as h5:
    # load intensity array
    raw_arr = h5["SpectraGroups/InitialMeasurement/spectra"][:]

# Normalizes spectra using normalize function
if not pre_norm == None:
    raw_arr = normalize(raw_arr, function = pre_norm)

# Sets pre_norm to "" for file name if None is selected
else:
    pre_norm = ""





# Normalizes the spectra on a scale of 0-1 depending on which normalization method was chosen
if normalization == "global":
    
    # max of all intensities
    arr_max = np.max(raw_arr)
    
    # normalization with respect to arr_max
    renorm_int = raw_arr/arr_max
    
elif normalization == "local":
    
    # each pixel is normalized independently
    for pixel in raw_arr:
        max_intensity = np.max(pixel)
        pixel /= max_intensity
    renorm_int = raw_arr
    
elif normalization == "localTIC":
    
    # each pixel is normalized independently
    for pixel in raw_arr:
        pixel /= np.sum(pixel)
    renorm_int = raw_arr
    
elif normalization == "globalTIC":
    
    # max TIC of all the spectra
    maxTIC = np.sum(raw_arr, axis = 1).max()
    
    # normalization with respect to the maxTIC
    renorm_int = raw_arr/maxTIC
 
    
 
    
# Define intensity threshold from 0-1
int_threshold = 0.11

# Convert int_threshold to 'str' to be used in file name
# Digits in str correspond to decimal places not ones place
# ex) 0.1 is converted to 10 
#     0.01 is converted to 01
# this is done to avoid file save errors with a decimal poin in the name
int_threshold_str = str(int_threshold).split(".")[1]


# for single digit thresholds (0.1, 0.2, 0.3, ..., 0.9) a 0 must be added after to keep two digits in place and avoid ambiguous thresholds in the file name
if int_threshold in np.arange(0,1,0.1).round(6):
    int_threshold_str = f"{int_threshold_str}0"


# Converts the normalized fingerprints to the binary fingerprints
# All intensities above the int_threshold are assigned a 1
# All intensities below the int_threshold are assigned a 0
total_fingerprints = np.where(renorm_int > int_threshold, 1, 0)

# file name notation is as follows
# {.sbd name}_{normalization method}_it{int_threshold}
# it is an acronym for intensity threshold to shorten file name
output_name = f"{name.split('.')[0]}_{str(pre_norm)}{normalization}_it{int_threshold_str}"

# Save binary fingerprints using .sbd file name, normalization method, and int_threshold as the save file name
np.save(f"{save_path}{output_name}", total_fingerprints)
