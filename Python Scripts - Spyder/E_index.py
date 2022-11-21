# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:44:04 2021

@author: nellin
"""

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

# Multiple E-index functions were tested and the best was found to be E_index()
# Results can vary and the E-index serves to point to where the most optimal paramters could be
# The final call for the most optimal parameters is up to the user 

# E-Index functions
def E_index(RR_dictionary, include, omit = [], E_type = "max", weight = "weight_PC"):
    
    # Function Description
    """
    
    Equations
    --------
    
    max : (max(s_low, s_high) - s_mid)/s_mid
    robust : ((s_low - s_mid) + (s_high - s_mid))/s_mid

    
    Arguments
    ----------
    RR_dictionary : dictionary
        DESCRIPTION : RR_dictionary from "binary_MultComp_IMS_data_PCA_fast.py" script
        format : Keys are each Principal Component; items within each key are 3 numpy arrays the correspond to the low, mid, and high region RR result for
        each C threshold in increasing order of C threshold
    
    include : int
        DESCRIPTION : Integer value for the total number of PCs to be used in the calculation.
    
    omit : list
        DESCRIPTION : List out the PCs that you wish to be omitted from the calculation as a lis of strings using the keys in the dictionary should the PC come before the value of include.
            Ex) if you would like to include = 5 PCs but you want to omit PC2. Then PCs 1 and 3-6 will be used in the calculation. 
            Default is an empty list.
        format : ['PC1', 'PC2', . . . 'PCn'] 
        
    E_type : str 
        DESCRIPTION : E-index formula to be used in the calculations. 'max' or 'robust'
    
    weight : str
        DESCRIPTION : weight function to be used in the calculations. 'weight_n', 'weight_PC', or 'fraction'
    
    Returns
    -------
    w_E : dictionary
        DESCRIPTION : dictionary of E-index values for each Principal Component. 
        format : Keys are each Principal Component; item in each key is the E-index value for the respective PC
    
    
    """
            
    # Creat E-index dictionary to be returned
    E = {}
    
    # Begin iterating across RR_dictioanry (input)
    for i, PC in enumerate(RR_dictionary):
        if PC in omit:
            pass
        else:
            if include > i-len(omit):
                
                # Define average similarity values for the low, mid, and high selected regions for PC
                s_low, s_mid, s_high = np.average(RR_dictionary[PC][0]), np.average(RR_dictionary[PC][1]), np.average(RR_dictionary[PC][2])
                
                if E_type == 'max':
                    # Calculate E value and insert into E dictionary
                    E[PC] = float(((max(s_low, s_high) - s_mid)/s_mid))
                    
                if E_type == 'robust':
                    # Calculate E value and insert into E dictionary
                    E[PC] = float(((s_low-s_mid) + (s_high-s_mid))/s_mid)
    
    if weight == 'weight_n':
        coefficient = 1/sum(E.values())
        # E_sqr = [e**2 for e in E.values()]
        E_sqr = [e*abs(e) for e in E.values()]
        E_sqr_sum = sum(E_sqr)
        w_E = coefficient*E_sqr_sum
    
    if weight != 'weight_n':
        if weight == 'fraction':
            W = [(1/include) for key in RR_dictionary]
        
        if weight == 'weight_PC':
            dof = include + 1
            W_denominator = [(dof-k) for k in range(1, dof)]
            W = [(dof - k)/sum(W_denominator) for k in range(1, dof)]
        
        w_E = sum([e*w for e,w in zip(E.values(), W)])

    # Return dictionary of E-index values
    return w_E


# Define folder contianing similarity results
folder = 'Results/04-17-2019_MB_Imaging_raw_data/RR_results/NAPCA-local/local it10/'

# Define save path for E index plots
save_path = 'Results/04-17-2019_MB_Imaging_raw_data/plots/E-index/'

# Create dictionaries for E index values and x values (selected pixel percentage)
xs = []
w_E = {}
w_E_r = {}
w_E_n = {}

# Iterate across folder 
for file in os.listdir(folder):
    
    # Open dictionary file
    dictionary = pkl.load(open(folder + file, "rb"))
    
    # Extract Selected pixel percent        
    rp = file.split("_")[-2]
    x = float(rp.replace("sp", "."))*100
    xs.append(float(rp.replace("sp", "."))*100)
    
    # Calculate E values with different parameters
    w_E[file] = E_index(dictionary, include = 5, omit = [], E_type = 'max', weight = 'weight_PC')
    w_E_n[file] = E_index(dictionary, include = 5, omit = [], E_type = 'max', weight = 'weight_n')
    w_E_r[file] = E_index(dictionary, include = 5, omit = [], E_type = 'robust', weight = 'weight_PC')

    
# Create nested dictionary containing all E index values            
ys = {'max':w_E,  'N_n':w_E_n, 'robust':w_E_r}

# List of colors for plots
color = ['b', 'grey', 'r']

# Create title for plots
title = folder.split('/')
title = (f'{title[1]}_{title[3]}')

# Iterate across nested dictionary to plot E values
for E_type, c in zip(ys,color):
    plt.figure(dpi=400)
    plt.scatter(xs, ys[E_type].values(), color = c, label = E_type)
    plt.title(f'{title}_{E_type}')
    plt.ylabel('N_index')
    plt.xlabel('Selected Pixel Percent')
    plt.grid(True)
    plt.legend(bbox_to_anchor = (1.25,1))
    plt.savefig(f'{save_path}{title}_{E_type}', bbox_inches = 'tight')


plt.show()
