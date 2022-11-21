# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:42:17 2021

@author: nellin
"""

# Imprt necessary modules
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os

# Define folder path for RR results from binary_MultComp_IMS_data_PCA_fast.py
folder = 'Results/2022-02-22_NE_MB_posMS_Lipids_DAN_spray/RR_results/NAPCA-local/local it03/'

# Define save path for 2D plots of RR results
save_path = 'Results/2022-02-22_NE_MB_posMS_Lipids_DAN_spray/plots/2D/NAPCA-local/local it03/'

# Iterate across results folder to pull file names
for file in os.listdir(folder):
       
    # Create file path
    file_path = folder + file
    
    # Load RR retult
    all_rr_dict = pkl.load(open(file_path, "rb"))
    
    # Pull intensity threshold and selected pixels percent from file name (must keep naming convention from previous scripts to avoid errors)
    params = file.split("it")[1]
    params = params.split("_")
    selected_pixels = int(params[-2].split("sp")[1])
    int_threshold = int(params[0])/100
       
    
    # Define label for x axis 
    labels = ['low','mid','high']
    
    # Create x points
    x = np.arange(len(labels))
    
    # Plot 2D plots by averaging the RR results of each region for each PC
    plt.figure(dpi = 400)
    
    # Iterate across each Principal component 
    for PC in all_rr_dict:
    
        # Create empty array
        region_avg = np.empty(0)
        
        # Iterate across the Principal components to pull each region
        for region in all_rr_dict[PC]:
        
            # Average the similarity values for the region
            region = np.average(region)
            
            # Append averaged value to empty array
            region_avg = np.append(region_avg, region)
        
        # Plot 
        plt.plot(x, region_avg, label = PC)
        plt.scatter(x, region_avg)
    
    # Create labels for each plot 
    # Recommended to follow the same naming convention to keep file names consistent 
    title = f"{file.split('.')[0]}_average"
    title = f"{title.split('_')[0]}_{'_'.join(title.split('_')[-6:])}"

    plt_title = f"Intensity threshold = {int_threshold} selected pixels = {selected_pixels}%"


    plt.title(plt_title, loc = "right")
    plt.xticks(x, labels)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.ylabel('RR Values')
    plt.xlabel('PCA Regions')
    plt.legend(bbox_to_anchor = (1.2,1))
    
    # Save plot
    plt.savefig(f'{save_path}{title}', bbox_inches = 'tight')
    plt.show()
