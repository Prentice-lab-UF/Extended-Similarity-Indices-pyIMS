# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:42:49 2021

@author: nellin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:29:08 2021

@author: nellin
"""
# Import necessary modules

import numpy as np
from math import ceil
import time
import pickle as pkl
import os
import matplotlib.pyplot as plt

# Define counters function to calculate each similarity counter used for all possible extended similarity indices. 
# a: total number of coincident ones greater than the coincident threshold
# d: total number of coincident zeros greater than the coincident threshold
# p: total length of single fingerprint
# total_dis: adds all counters less than the coincident threshold
# total_sim: adds all counters greater than the coincident threshold
# weighted counters were calculated for all of the abose and are noted with a 'w' in the dictionary keys

# data_sets = 1-D array of the binary data matrix selected for calculation summed across axi-0 and the total number of fingerprints at index [-1]
# c_threshold = threshold with which the counters are compared to determine of counted as a 1-similarity 0-similarity or disimilarity. Possible
#       values can range from nmod2 to n-1 where n is equal to the total number of fingerprints and mmd od is modulus 
# w_factor = the type of weighted function that will be applied to the counters. Options are power, fraction, or none. 

def calculate_counters(data_sets, c_threshold=None, w_factor="fraction"):
    # Setting matches
    total_data = np.sum(data_sets, axis=0)
    n_fingerprints = total_data[-1]
    c_total = total_data[:-1]
    matches = (n_fingerprints + 1) * [0]
    for i in range(n_fingerprints + 1):
        matches[i] = np.count_nonzero(c_total == i)
    matches = np.array(matches)
    
    # Assign c_threshold
    if not c_threshold:
        c_threshold = n_fingerprints % 2
    if isinstance(c_threshold, str):
        if c_threshold != 'dissimilar':
            raise TypeError("c_threshold must be None, 'dissimilar', or an integer.")
        else:
            c_threshold = ceil(n_fingerprints / 2)
    if isinstance(c_threshold, int):
        if c_threshold >= n_fingerprints:
            raise ValueError("c_threshold cannot be equal or greater than n_fingerprints.")
        c_threshold = c_threshold
    
    # Calculate d_vector
    d_vector = np.array([abs(2 * k - n_fingerprints) for k in
                        range(n_fingerprints + 1)])
    
    # Set w_factor
    if w_factor:
        if "power" in w_factor:
            power = int(w_factor.split("_")[-1])
            def f_s(d):
                return power**-float(n_fingerprints - d)
    
            def f_d(d):
                return power**-float(d - n_fingerprints % 2)
        elif w_factor == "fraction":
            def f_s(d):
                return d/n_fingerprints
    
            def f_d(d):
                return 1 - (d - n_fingerprints % 2)/n_fingerprints
        else:
            def f_s(d):
                return 1
    
            def f_d(d):
                return 1
    else:
        def f_s(d):
            return 1
    
        def f_d(d):
            return 1
    weights = (n_fingerprints + 1) * [0]
    for k in range(n_fingerprints + 1):
        if d_vector[k] > c_threshold:
            weights[k] = f_s(d_vector[k])
        else:
            weights[k] = f_d(d_vector[k])
    weights = np.array(weights)
    
    # Set weighted matches
    weighted_matches = matches * weights
    
    # Calculate a, d, b+c
    a = 0
    w_a = 0
    d = 0
    w_d = 0
    total_dis = 0
    total_w_dis = 0
    for k in range(n_fingerprints + 1):
        if 2 * k - n_fingerprints > c_threshold:
            a += matches[k]
            w_a += weighted_matches[k]
        elif n_fingerprints - 2 * k > c_threshold:
            d += matches[k]
            w_d += weighted_matches[k]
        else:
            total_dis += matches[k]
            total_w_dis += weighted_matches[k]
    total_sim = a + d
    total_w_sim = w_a + w_d
    p = total_sim + total_dis
    w_p = total_w_sim + total_w_dis
    
    counters = {"a": a, "w_a": w_a, "d": d, "w_d": w_d,
                "total_sim": total_sim, "total_w_sim": total_w_sim,
                "total_dis": total_dis, "total_w_dis": total_w_dis,
                "p": p, "w_p": w_p}
    
    return counters


# Start timer for calculaitons and data importation
start = time.time()

print("Loading total fingerprints. . .")

# Define file name of all spectra of the image
total_pixels_file = '04-17-2019_MB_Imaging_raw_data_RMSlocal_it10.npy'

# Define save path for simillarity save files of each set of selected pixels
save_path = 'Results/04-17-2019_MB_Imaging_raw_data/RR_results/'

# Define file path for total_pixels_file
total_pixels_path = 'Fingerprint_data/04-17-2019_MB_Imaging/total_fingerprints/'

# Define folder path for selectd pixels files (must be the folder not an individual file)
selected_fp_folder = 'Fingerprint_data/04-17-2019_MB_Imaging/selected_pixels/pixels_loop/'

# Load in total pixels dataset and set data type to 'int8'
total_pixels = np.load(total_pixels_path + total_pixels_file)
total_pixels = np.array(total_pixels, dtype = 'int8')

# In this script the Russel-Rao similarity index is being used to calculate similarity.
# To speed up the run time of the script the columns of the dataset that sum to 0 (meaning there are zero coincident ones) are deleted while saving the original total fingerprint length.
# This does not change the Russel-Rao similarity values calculated but does make the run time significantly shorter for large datasets with many 0s due to less columns having to be iterated over.
p = len(total_pixels[0,:])
summed = np.sum(total_pixels, axis=0)
zeros = np.array(np.where(summed==0))
total_pixels = np.delete(total_pixels, zeros, axis=1)

# Starts for loop using the sleected pixels folder
# The for loop opens the folder and selects the first selected pixels file to pick the pixels of interest from the total dataset
# and continues until all folders have been used to calculate the similarity values
for file in os.listdir(selected_fp_folder):
    
    # Combines folder path and file name to create the file path of the file of interest
    file_path = selected_fp_folder + file

    # Prtins name of file being used in order to monitor how far along the script is
    print(file)
    
    # Load in selected pixels file to be used for calculation
    pixel_indices = pkl.load(open(file_path, "rb"))
    
    # Define the Russel-Rao dictionary where the similarity values will be stored for each principal of the PCA
    rr_dict = {}

    # Starts for loop using the selected pixels dictionary. Each key corresponds to a principal component of the PCA      
    for key in pixel_indices:
        # print key to know how many PCs are left in the file to calculate the similarity
        print(key)
        
        # for loop to caclulate the similarity of each region of the principal component in order from low to high scores values regions
        for array in pixel_indices[key]:
            
            # find total number of fingerprints in the region to be calculated
            fp_total = len(array)
    
            # Define 20 coincidence thresholds from 0-100% of the total number of fingerprints in the region
            # The 20 coincidence thresholds are in ~5% increments 
            # The threshold at ~100% is equal to n-1
            c_thresh = np.linspace(fp_total%2, fp_total, 20)
            c_thresh[-1] = c_thresh[-1] - 1
            
            # Define array for calculated RR similarity values for region x of PCy at increasing coincidence thresholds
            rr_arr = np.empty([0])
                    
            # Select fingerprints for comparison from the total dataset
            total_fingerprints = np.array(total_pixels[array], dtype = 'int8')
            
            # Sum selected fingerprints column wise
            condensed_fingerprints = np.sum(total_fingerprints, axis=0)
            
            # Append total fingerprints to the end of the condensed array at index [-1]
            data_sets = np.array([np.append(condensed_fingerprints, fp_total)])
            
            # Begins calculations of similarity counters for each coincidence threshold defined in the c_thresh array
            # and calculates the Russel-Rao similarity to append to the rr_arr
            for c in c_thresh:
                counters = calculate_counters(data_sets, c_threshold = c)
                rr = counters["w_a"]/(p)
                rr_arr = np.append(rr_arr, [rr], axis = 0)
                
            # Creates key for rr_dict if key is not present (key corresponds to the key of the selected pixels dictionary) and adds the rr_arr as a list to the dictionary
            # if the key is present then the rr_arr is simply added to the key as a list
            if key not in rr_dict:
                rr_dict[key] = list([rr_arr])
            else: 
                rr_dict[key].append(rr_arr)
                            
    
    # Create save file name by splitting total_pixel file name and selected pixels file name and combining the elements into a list
    output = (f"{total_pixels_file.split('.')[0]}_{file}").split("_")
    
    # Deletes extraneous information in the output file name list to make it more concise. indices will vary depending on the file convention used by the user and may notneed to be used at all
    del output[8:-1]
    
    # Rejoins all elements of the file name to create final save file name
    output = "_".join(output) + "_RR"
    
    # Tries to save the file with the output file name and save path defined earlier using pickle module
    # If file fails to save "File not saved" is returned so the user is aware
    try: 
        out_file  = open(f"{save_path}{output}_LL", "wb")
        pkl.dump(rr_dict, out_file)
        out_file.close()
    except:
        print("File not saved")
       
# Calculates the final calculation time of the script and prints it in the console in minutes
Calculation_time = time.time() - start
print('Calculation time =', Calculation_time/60,'min')
             