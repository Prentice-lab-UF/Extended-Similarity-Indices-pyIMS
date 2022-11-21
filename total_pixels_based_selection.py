# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:51:46 2021

@author: nellin
"""

# Import necessary modules
import numpy as np
import pickle as pkl

# Define select_pixels function
# select pixels based on the scores from PCA for each principal component and saves selected pixels as numpy array
def select_pixels(array, percent):
    
    # Funciton description
    """
    
    Arguments
    ----------
    array : 'numpy array'
        DESCRIPTION : .npy file that contians the scores values and pixel indices 
        format : (p, n, 2); p = the number of principal components; n = the total number of pixels in the image; 
            column-0 = index for each pixel; column-1 = scores for respective pixel
    
    percent : 'float' 
        DESCRIPTION : define percentage of the total pixels in the image to be selected for comparison
            values should not exceed 33% otherwise pixels selected will converge and overlap with other regions
        format : 'float' decimal values in range 0-1 with 1 being 100% of the total pixels in the image 

    Returns
    -------
    region_dict : 

    """
    
    # Find the number of pixels to be selected based on the percentage chosen and the total number of pixels selected
    selected_perc = int(percent*len(array[0,:,0]))
    
    # Create dictionary to store values for each PC and each region
    region_dict = {}
    
    # Iterate across the input array to sort the pixels based on the scores values from least to greated with least at index [0,:]
    for PC, arr in enumerate(array):
        
        # Create String for key in dictionary 
        PC_str = f"PC{str(PC+1)}"
        
        # Sort pixels for each PC based on the score values in column 0 and keeping their respective indices in column 1
        sorted_scores = arr[arr[:, 1].argsort()]
        
        # Sort the pixels based on the absolute score values from least to greates
        # Closest to zero is at index [0,:]
        abs_sorted = abs(sorted_scores)[abs(sorted_scores)[:,1].argsort()]
        
        # Select % pixels with the most negative scores
        low_pixels = np.array(sorted_scores[:selected_perc, 0], dtype = 'int64')
        
        # Select % pixels with the most positive scores
        high_pixels = np.array(sorted_scores[-selected_perc:, 0], dtype = 'int64')
        
        # Select % pixels closest to zero
        mid_pixels = np.array(abs_sorted[:selected_perc, 0], dtype = 'int64')
        
        # Create key for the respective PC and bundles the three regions into a nested list
        region_dict[PC_str] = list([low_pixels, mid_pixels, high_pixels])
    
    # Returns the complete dictionary with each PC as a key and items as a list of the three regions
    # Each region is a 1D array with the indices of the selected pixels
    return region_dict
    
# Define Scores numpy file from "CSV_scores_loadings-npy_converter.py"
scores_file = 'PCA_04-17-2019_MB_Imaging_raw_data-Scores.npy'

# Define folder paths for {scores_file}
folder = 'Fingerprint_data/04-17-2019_MB_Imaging/scores/'

# Define the save path to be used to save the output dictionaries
save_path = 'Fingerprint_data/04-17-2019_MB_Imaging/selected_pixels/pixels_loop/'

# Create array of percentages between 0 and 1 to be used in the select_pixels() function
percent_arr = np.arange(0.01, 0.31, 0.01).round(6)

# Load numpy scores file to be used in select_pixels() function
scores = np.load(f"{folder}{scores_file}")

# Iterate across {percent_arr} to create multiple dictionaries of different select percentages
for percent in percent_arr:
    
    # Convert the percentage to a string argument to be used for the file name 
    percent_str = str(percent).split(".")[1]
    
    # Run select_pixels() function 
    selected_pixels = select_pixels(scores, percent)
    
    # Convert single digit tenths float to two character string 
    if percent in np.arange(0,1,0.1,).round(6):
        percent_str = f"{percent_str}0"
    
    # Define output file name/path
    output = f"{save_path}{scores_file.split('.')[0]}_sp{percent_str}"
    
    # Save dictionary using pickle
    # If file fails to save {"Save Failed"} prints in console
    try: 
        out_file  = open(output, 'wb')
        pkl.dump(selected_pixels, out_file)
        out_file.close()
    except:
        print("Save Failed")
    
