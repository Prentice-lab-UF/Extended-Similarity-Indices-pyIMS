# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 18:18:26 2021

@author: nellin
"""
# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

# Define 'folder' as the folder path containing the results dictionaries output from 'binary_MultComp_IMS_data_PCA_fast.py'
# and the save path where the 3D plots will be saved 
folder = 'Results/04-17-2019_MB_Imaging_raw_data/RR_results/NAPCA-local/local it08/'
save_path = 'Results/04-17-2019_MB_Imaging_raw_data/plots/3D/NAPCA-local/local it08/'

# Create a list of the file names in the results folder defined in line 15 then iterating over the list to work with each dictionary individually 
for file in os.listdir(folder):
    
    # Combine folder and dictionary file name to create file path
    file_path = folder + file
    
    # Define x labels of the 3D plot
    labels = ['low', 'mid', 'high']
    
    # Define the z coordinates of the 3D plot.
    # Each element within 'perc_c' corresponds to the coincidence threshold used for each calculation as a percentage of 
    #   the total number of pixels that make up the image
    # Increments of 5% were chosen as a default starting at ~0 and ending at ~100 
    # (not truly 100% since the coincidence threshold cannot equal the total number of pixels in the image, 
    #   instead the 100% means the coincidence threshold is at n-1 where n is the total number of pixels)
    # Will also be used to iterate over the corresponding similarity results
    perc_c = np.arange(0, 100, 5)
    
    # Define numerical x coordinates to be used based off the labels since matplotlib.pyplot must have real numbers to be able to plot points
    xs = np.arange(len(labels))
    
    # Define and open the Russel-Rao results dicitonary
    all_rr_dict = pkl.load(open(file_path, "rb"))
    
    # Pull intensity threshold and selected pixels percent from file name (must keep naming convention from previous scripts to avoid errors)
    params = file.split("it")[1]
    params = params.split("_")
    selected_pixels = int(params[-2].split("sp")[1])
    int_threshold = int(params[0])/100

    
    
    # Iterate of the keys of the results dictionary that correspond to each principal component of the PCA
    for PC in all_rr_dict:
        
        # Begin plotting the data
        # Must define fig based on the principal component being used so every new principal component a new figure is created
        fig = plt.figure(PC, dpi = 500)
        
        # Set to 3-D plot wit x, y, and, z axes
        ax = plt.axes(projection = '3d')
        
        # For loop to select data points based on the coincidence threshold 
        for cx, c in enumerate(perc_c):
            
            # Organize selected points for individula coincidence threshold bein iterated
            # Reshaped to fit formate of plot and scatter functions and multiplied by 10000 to make axis labels more legible (magnitude is indicated in axis label)
            lmh = np.array([all_rr_dict[PC][0][cx],all_rr_dict[PC][1][cx],all_rr_dict[PC][2][cx]]).reshape(3)*10000
            
            # plots data points in 3D format
            # x-axis = region label (low, mid, high)
            # y-axis = % coincidence threshold corresponding to perc_c
            # z-axis = similarity coefficients (z-axis is the vertical axis)
            ax.plot(xs, c.repeat(3),lmh)
            ax.scatter(xs, c.repeat(3), lmh)
        
        # Format 3D plot
        
        plt_title = f"{PC} Intensity threshold = {int_threshold} selected pixels = {selected_pixels}%"
        plt.title(plt_title, loc = "right")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Scores Regions")
        ax.set_ylabel("% C Threshold")
        ax.set_zlabel("RR Value (x10$^-$$^4$)")
        
        # Adjust viewing angle of 3D plot
        ax.view_init(15, 45)
        
        # Save plot as figure
        plt.savefig(save_path + file.split(".")[0] + "_" + PC, dpi = 800)
        plt.show()
            