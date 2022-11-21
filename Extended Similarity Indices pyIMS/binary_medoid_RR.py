# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:15:21 2021

@author: nellin
"""

# Import necessary modules 
import numpy as np
import pickle as pkl
from math import ceil
import time 
import os

# Define counters function
def calculate_counters(data_sets, c_threshold=None, w_factor="fraction"):
    # Setting matches
    total_data = np.sum(data_sets, axis=0)
    n_fingerprints = int(total_data[-1])
    c_total = total_data[:-1]
    
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
    
    # Calculate a, d, b+c
    a = 0
    w_a = 0
    d = 0
    w_d = 0
    total_dis = 0
    total_w_dis = 0
    for s in c_total:
        if 2 * s - n_fingerprints > c_threshold:
            a += 1
            w_a += f_s(2 * s - n_fingerprints)
            #print(f_s(2 * s - n_fingerprints))
        elif n_fingerprints - 2 * s > c_threshold:
            d += 1
            w_d += f_s(abs(2 * s - n_fingerprints))
        else:
            total_dis += 1
            total_w_dis += f_d(abs(2 * s - n_fingerprints))
    total_sim = a + d
    total_w_sim = w_a + w_d
    p = total_sim + total_dis
    w_p = total_w_sim + total_w_dis
    #print(w_a, w_d, p)
    #print((w_a+w_d)/p)
    
    counters = {"a": a, "w_a": w_a, "d": d, "w_d": w_d,
                "total_sim": total_sim, "total_w_sim": total_w_sim,
                "total_dis": total_dis, "total_w_dis": total_w_dis,
                "p": p, "w_p": w_p}
    
    return counters
    
# Define start time to record the run time of the script 
start = time.time()

# Load in Binary MS data matrix
total_pixels_file = "2022-02-22_NE_MB_posMS_Lipids_DAN_spray_local_it01.npy"
total_pixels_path = "/blue/rmirandaquintana/nellin/Fingerprint_data/2022-02-22_NE_MB_posMS_Lipids_DAN_spray/total_fingerprints/"
total_data = np.load(total_pixels_path+total_pixels_file)
total_data = np.array(total_data, dtype = 'int8')

# Load in selected pixels file
file = "2022-02-22_NE_MB_posMS_Lipids_DAN_spray_None_PCA-5-Scores_sp01"
selected_pixels_folder = "/blue/rmirandaquintana/nellin/Fingerprint_data/2022-02-22_NE_MB_posMS_Lipids_DAN_spray/selected_pixels/None-5/"
pixel_indices = pkl.load(open(selected_pixels_folder + file, "rb"))

# Define save path for medoid spectra indices 
save_path = "/blue/rmirandaquintana/nellin/Results/2022-02-22_NE_MB_posMS_Lipids_DAN_spray/"

# Create medoid indices dictionary and corresponding RR dictionary 
# RR values are the similarity values that result from the removal of the corresonding binary fingerprint (spectrum)
index_dict = {}
rr_dict = {}

# Iterate across each PC and each region within the PC (low, mid, and high)
for PC in pixel_indices:
    for r,region in enumerate(pixel_indices[PC]):
    
        # Convert region into numpy array
        region = np.array(region)
        
        # Print the PC and region (as an integer) to keep track of the scripts progression since the script can take a significant amount of time
        print(PC, r)
        
        region_size = np.size(region)
        
        # Calculate the Medoid spectra until it is narrowed down to 5 or less spectra 
        while region_size > 5:
            
            
            print("loop")
            print(len(region))
            
            # Create matrix of selected binary fingerprints (pixels) to be used in medoid calculations
            selected_pixels = total_data[region]
            
            # Create temp list for Russell-Rao similarity values
            rr_list = []
            
            # number of points in the calculation
            total_points = len(selected_pixels)
    
            # vector of column sums
            total_sum = np.sum(selected_pixels, axis = 0)
            min_sim = total_points
            
            # Iterate across the selected pixel matrix
            for i, pixel in enumerate(selected_pixels):
                
                # Sum of the selected data matrix without the iterated pixel
                i_sum = total_sum - selected_pixels[i]
                
                # Create data set for counters function
                data_sets=np.array([np.append(i_sum, total_points - 1)])
 
                # Calculate similarity counters
                counters = calculate_counters(data_sets, c_threshold=None)
                
                # Calculate the Russell-Rao similarity 
                rr_nw = (counters['w_a'])/(counters['p'])
                
                # Append RR similarity and pixel indices to temp RR and index lists
                rr_list.append(rr_nw)
            
            # Convet RR list to numpy array
            rr_list = np.array(rr_list)
            
            # Select medoid pixel indices based on the minimum RR value and redefine as region variable
            region = region[np.where(rr_list == np.min(rr_list))]
                        
            # Save the minimum RR value
            rr_list = np.min(rr_list)
                        
            if region_size == np.size(region):
                print(f"Less than 5 medoid spectra not solvable for {PC}. Appending all equivalent spectra indices to indices dictionary")
                region_size = 1
            else:
                region_size = np.size(region)

        print(region)
        # Add RR value and medoid pixel indices to respective dictionary for iterated PC  
        if not PC in index_dict:
            rr_dict[PC] = list([rr_list])
            index_dict[PC] = list([region])
        else:
            index_dict[PC].append(region)
            rr_dict[PC].append(rr_list)
    
# Create output file names
out1 = total_pixels_file.split(".")[0]
out2 = file.split("_")[-3:]
out2 = "_".join(out2)
out_medoid_filename = f'{out1}_{out2}_medoid-spectra'
out_rr_filename = f'{out1}_{out2}_medoid_RR_values'

# Save dictioanries as binary files
try: 
    out_medoid_file  = open(f'{save_path}{out_medoid_filename}', 'wb')
    out_rr_file = open(f'{save_path}{out_rr_filename}', 'wb')
    pkl.dump(index_dict, out_medoid_file)
    pkl.dump(rr_dict, out_rr_file)    
    out_medoid_file.close()
    out_rr_file.close()
except:
    print("File not saved")

# Final run time of Script
finish = time.time() - start
print(finish)
# selected_pixels[index] is going to be the most representative fingerprint