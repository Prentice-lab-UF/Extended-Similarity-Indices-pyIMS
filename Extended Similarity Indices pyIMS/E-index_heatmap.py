# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 19:31:16 2022

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
        DESCRIPTION : weight function to be used in the calculations. 'squared_sum', 'weight_PC', or 'fraction'
    
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

    
    if weight == 'squared_sum':
        coefficient = 1/sum(E.values())
        # E_sqr = [e**2 for e in E.values()]
        E_sqr = [e*abs(e) for e in E.values()]
        E_sqr_sum = sum(E_sqr)
        w_E = coefficient*E_sqr_sum
    
    if weight != 'squared_sum':
        if weight == 'fraction':
            W = [(1/include) for key in RR_dictionary]
        
        if weight == 'weight_PC':
            dof = include + 1
            W_denominator = [(dof-k) for k in range(1, dof)]
            W = [(dof - k)/sum(W_denominator) for k in range(1, dof)]
        
        w_E = sum([e*w for e,w in zip(E.values(), W)])

    # Return dictionary of E-index values
    return w_E



# Define folder contianing similarity results for specific normalization (i.e. NAPCA-local)
folder = 'results/2022-02-22/RMS-RMSlocal PCA/'

# Define save path for E index plots
save_path = 'results/E-index plots/2022-02-22/RMS-RMSlocal PCA/'

# Create dictionaries for E index values and x values (selected pixel percentage)
ys = []
max_dict = {}
max_sq_dict = {}
max_f_dict = {}
robust_dict = {}
robust_sq_dict = {}
robust_f_dict = {}


# Open folder containing RR of all intensity thresholds tested for normalization method 
# Begin iteration of opening intensity threshold folder starting at lowest it tested
# Recommend testing 0.01-0.20 to limit errors
for it_subfolder in os.listdir(folder):
    
    # get intensity threshold from it_subfolder namer
    it = it_subfolder.split("it")[-1]
    ys.append(int(it))
    
    # Create temorarpy dictionaries for E-index values calculated for intensity threshold
    w_E = {}
    w_E_r = {}
    w_E_n = {}
    w_E_rn = {}
    w_E_f = {}
    w_E_rf = {}

    # Create folder path for dictionary files in it_subfolder
    it_subfolder_path = folder + it_subfolder + '/'
    
    # Create list fo x values used for plotting 
    xs = []
    
    print(it_subfolder)
    
    # Iterate across it_subfolder to open dictionary files 
    for file in os.listdir(it_subfolder_path):
       
        # Open dictionary file
        dictionary = pkl.load(open(it_subfolder_path + file, "rb"))
        
        # Extract Selected pixel percent to be used as x values for plotting
        # based on file names so naming convention must not have been altered from previous scripts
        rp = file.split("_")[-2]
        x = float(rp.replace("sp", "."))*100
        xs.append(float(rp.replace("sp", "."))*100)
        
        # Calculate E values with different parameters
        w_E[file] = E_index(dictionary, include = 5, omit = ['PC2'], E_type = 'max', weight = 'weight_PC')
        w_E_n[file] = E_index(dictionary, include = 5, omit = ['PC2'], E_type = 'max', weight = 'squared_sum')
        w_E_f[file] = E_index(dictionary, include = 5, omit = ['PC2'], E_type = 'max', weight = 'fraction')
        w_E_r[file] = E_index(dictionary, include = 5, omit = ['PC2'], E_type = 'robust', weight = 'weight_PC')
        w_E_rn[file] = E_index(dictionary, include = 5, omit = ['PC2'], E_type = 'robust', weight = 'squared_sum')
        w_E_rf[file] = E_index(dictionary, include = 5, omit = ['PC2'], E_type = 'robust', weight = 'fraction')


    # Append temporary E-index dictionaries to parent dictionary that will be usedto plot all values
    max_dict[it_subfolder] = w_E
    max_sq_dict[it_subfolder] = w_E_n
    max_f_dict[it_subfolder] = w_E_f
    robust_dict[it_subfolder] = w_E_r
    robust_sq_dict[it_subfolder] = w_E_rn
    robust_f_dict[it_subfolder] = w_E_rf


# Create dictionary containing all parent dictionaries
all_E = {'max_wPC':max_dict,  'max_wsq':max_sq_dict, 'max_wf':max_f_dict, 'robust_wPC':robust_dict, 'robust_wsq':robust_sq_dict, 'robust_wf':robust_f_dict}


# Create title for plots
title = folder.split('/')
title = (f'{title[1]}')

Z = np.zeros([6, len(ys),len(xs)])
print(np.shape(Z))

# iterate across all_E dictionary to plot all 6 tested E-index values as heatmaps.
for i,E_type in enumerate(all_E):
    for it_subfolder, y in zip(all_E[E_type], ys):
        Z[i,y-1,:] = np.array(list(all_E[E_type][it_subfolder].values()))
    
    fig, ax = plt.subplots(figsize=(7.5,5), constrained_layout=True)
        
    psm = ax.pcolormesh(Z[i,:,:], cmap='gist_ncar', rasterized=True)
    fig.colorbar(psm, ax=ax, aspect = 50)
    plt.title(f'{E_type}', pad = 20)
    ax.set_ylabel('Intensity Threshold', fontsize = 15, labelpad = 10)
    ax.set_xlabel('Selected Pixel Percent', fontsize = 15, labelpad = 10)
    ax.set_yticks([0,5,10,15,20])
    ax.set_yticklabels([0,0.05,0.10,0.15,0.20], fontsize = 15)
    ax.set_xticklabels([0,5,10,15,20,25,30], fontsize = 15)
    ax.set_xticks([0,5,10,15,20,25,30])

    plt.savefig(f'{save_path}{title}_{E_type}-heatmaps', bbox_inches = 'tight')
    plt.show()
